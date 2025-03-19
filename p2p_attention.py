import cv2
import abc
import torch
import numpy as np
from PIL import Image
from typing import List


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        # return self.num_att_layers if LOW_RESOURCE else 0
        return -1
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        # if self.cur_att_layer >= self.num_uncond_att_layers:
        #     if LOW_RESOURCE:
        #         attn = self.forward(attn, is_cross, place_in_unet)
        #     else:
        #         h = attn.shape[0]
        #         attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        

        if self.activate:
            # Avoid inplace modification
            # But do not feed conditional and unconditional together
            attn = self.forward(attn, is_cross, place_in_unet)
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

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
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
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


# =========================================================================================================

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        
        # https://github.com/google/prompt-to-prompt/issues/44#issuecomment-1593284782
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            x = hidden_states
            context = encoder_hidden_states
            mask = attention_mask
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)
            return to_out(out)
        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count



def aggregate_attention(prompts, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def get_cross_attetnion_maps(batch_size, attention_store: AttentionStore, res: int = 16, from_where: List[str] = ("up", "down")):
    with torch.no_grad():
        attention_maps = [
            aggregate_attention([""] * batch_size, attention_store, res, from_where, True, i)
            for i in range(batch_size)
        ]

        MAX_TOKEN_LEN = 77
        all_images = []
        for i in range(batch_size):
            images = []
            for j in range(MAX_TOKEN_LEN):
                image = attention_maps[i][:, :, j]
                image = 255 * image / image.max()
                image = image.unsqueeze(-1).expand(*image.shape, 3)
                image = image.numpy().astype(np.uint8)
                image = np.array(Image.fromarray(image).resize((256, 256)))
                images.append(image)
            attn_map = np.stack(images, axis=0)
            all_images.append(attn_map)
        all_images = np.stack(all_images, axis=0)
    return all_images


def otsu_thresholding(image: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's thresholding to a color image by converting it to grayscale.
    Parameters: image (np.ndarray): Input color image of shape (H, W, C). 
    Returns: np.ndarray: Binary thresholded image.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return np.expand_dims(binary_image, axis=2)



def image2latent(vae, image):
    """
    Encodes an image into latent space using the VAE encoder.
    
    Args:
        vae: The variational autoencoder (VAE) model.
        image: A numpy array representing the image, with shape (B, H, W, C) and values in [0, 255].
    
    Returns:
        latent: The encoded latent representation.
    """
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)#, dtype=torch.float16) 
    image = image / 255.0  # Normalize to [0, 1]
    image = (image - 0.5) * 2  # Scale to [-1, 1]
    image = image.permute(0, 3, 1, 2).to(vae.device)  # Convert to (B, C, H, W)
    
    with torch.no_grad():
        latent = vae.encode(image)['latent_dist'].mean  # Get latent distribution mean
    
    latent = latent * vae.config.scaling_factor  # Scale factor used in diffusion models
    return latent.detach().cpu().numpy()


def latent2image(vae, latent):
    """
    Decodes a latent into image space using the VAE decoder.
    
    Args:
        vae: The variational autoencoder (VAE) model.
        latent: A numpy array representing the latent, with shape (B, H//8, W//8, C).
        
    Returns:
        image: The decoded image representation.
    """
    if not isinstance(latent, torch.Tensor):
        latent = torch.tensor(latent)
    latent = latent.to(vae.device)
    latent = 1 / vae.config.scaling_factor * latent # 0.18215
    image = vae.decode(latent)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image