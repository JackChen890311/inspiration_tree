import abc
import torch
from typing import List
from diffusers.models.cross_attention import CrossAttention


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if not self.activate:
            return attn
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            # attn[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet)
            # Avoid in-place modification by concatenating the unchanged and modified parts
            attn = torch.cat([attn[:h // 2], self.forward(attn[h // 2:], is_cross, place_in_unet)], dim=0)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.activate = True
    
    def switch(self):
        self.activate = not self.activate


class EmptyControl(AttentionControl):
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # if attn.shape[1] <= 32**2:
        self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class P2PCrossAttnProcessor:
    def __init__(self, controller, place_in_unet):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # one line change
        self.controller(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def register_attention_control(unet, controller):
    attn_procs = {}
    cross_att_count = 0
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[
                block_id
            ]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue
        cross_att_count += 1
        attn_procs[name] = P2PCrossAttnProcessor(
            controller=controller, place_in_unet=place_in_unet
        )

    unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count


def get_average_attention(controller):
    average_attention = {
        key: [
            item / controller.cur_step
            for item in controller.attention_store[key]
        ]
        for key in controller.attention_store
    }
    return average_attention


def aggregate_attention(
    controller, bsz, res: int, from_where: List[str], is_cross: bool, select: int
):
    out = []
    attention_maps = get_average_attention(controller)
    num_pixels = res**2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(
                    bsz, -1, res, res, item.shape[-1]
                )[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out


def aggregate_attention_batched(
    controller, bsz, res: int, from_where: List[str], is_cross: bool
):
    out = []
    attention_maps = get_average_attention(controller)
    num_pixels = res**2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(
                    bsz, -1, res, res, item.shape[-1]
                )
                out.append(cross_maps)
    out = torch.cat(out, dim=1)
    out = out.sum(1) / out.shape[1]
    return out


# @torch.no_grad()
# def perform_full_inference(pipe, instance_prompt, path, guidance_scale=7.5):
#     unet = pipe.unet
#     vae = pipe.vae
#     text_encoder = pipe.text_encoder
#     tokenizer = pipe.tokenizer
#     device = pipe.device
#     validation_scheduler = pipe.scheduler
#     weight_dtype = pipe.weight_dtype

#     unet.eval()
#     text_encoder.eval()

#     latents = torch.randn((1, 4, 64, 64), device=device)
#     uncond_input = tokenizer(
#         [""],
#         padding="max_length",
#         max_length=tokenizer.model_max_length,
#         return_tensors="pt",
#     ).to(device)
#     input_ids = tokenizer(
#         [instance_prompt],
#         padding="max_length",
#         truncation=True,
#         max_length=tokenizer.model_max_length,
#         return_tensors="pt",
#     ).input_ids.to(device)
#     cond_embeddings = text_encoder(input_ids)[0]
#     uncond_embeddings = text_encoder(uncond_input.input_ids)[0]
#     text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

#     for t in validation_scheduler.timesteps:
#         latent_model_input = torch.cat([latents] * 2)
#         latent_model_input = validation_scheduler.scale_model_input(
#             latent_model_input, timestep=t
#         )

#         pred = unet(
#             latent_model_input, t, encoder_hidden_states=text_embeddings
#         )
#         noise_pred = pred.sample

#         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#         noise_pred = noise_pred_uncond + guidance_scale * (
#             noise_pred_text - noise_pred_uncond
#         )

#         latents = validation_scheduler.step(noise_pred, t, latents).prev_sample
#     latents = 1 / 0.18215 * latents

#     images = vae.decode(latents.to(weight_dtype)).sample
#     images = (images / 2 + 0.5).clamp(0, 1)
#     images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
#     images = (images * 255).round().astype("uint8")

#     Image.fromarray(images[0]).save(path)


# def text_under_image(
#     image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)
# ) -> np.ndarray:
#     h, w, c = image.shape
#     offset = int(h * 0.2)
#     img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     img[:h] = image
#     textsize = cv2.getTextSize(text, font, 1, 2)[0]
#     text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
#     cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
#     return img


# def view_images(
#     images: Union[np.ndarray, List],
#     num_rows: int = 1,
#     offset_ratio: float = 0.02,
#     display_image: bool = True,
# ) -> Image.Image:
#     """Displays a list of images in a grid."""
#     if type(images) is list:
#         num_empty = len(images) % num_rows
#     elif images.ndim == 4:
#         num_empty = images.shape[0] % num_rows
#     else:
#         images = [images]
#         num_empty = 0

#     empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
#     images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
#     num_items = len(images)

#     h, w, c = images[0].shape
#     offset = int(h * offset_ratio)
#     num_cols = num_items // num_rows
#     image_ = (
#         np.ones(
#             (
#                 h * num_rows + offset * (num_rows - 1),
#                 w * num_cols + offset * (num_cols - 1),
#                 3,
#             ),
#             dtype=np.uint8,
#         )
#         * 255
#     )
#     for i in range(num_rows):
#         for j in range(num_cols):
#             image_[
#                 i * (h + offset) : i * (h + offset) + h :,
#                 j * (w + offset) : j * (w + offset) + w,
#             ] = images[i * num_cols + j]

#     pil_img = Image.fromarray(image_)

#     return pil_img


# @torch.no_grad()
# def save_cross_attention_vis(self, prompt, attention_maps, path):
#     tokens = self.tokenizer.encode(prompt)
#     images = []
#     for i in range(len(tokens)):
#         image = attention_maps[:, :, i]
#         image = 255 * image / image.max()
#         image = image.unsqueeze(-1).expand(*image.shape, 3)
#         image = image.numpy().astype(np.uint8)
#         image = np.array(Image.fromarray(image).resize((256, 256)))
#         image = text_under_image(
#             image, self.tokenizer.decode(int(tokens[i]))
#         )
#         images.append(image)
#     vis = view_images(np.stack(images, axis=0))
#     vis.save(path)
