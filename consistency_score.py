"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
import os
import math
import torch
from diffusers import StableDiffusionPipeline
from IPython.display import clear_output
from transformers import CLIPImageProcessor, CLIPModel
from PIL import Image
import re
import argparse
import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_new_tokens", type=str, help="Path to directory with the new embeddings")
    parser.add_argument("--node", type=str, help="Path to directory with the new embeddings")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--model_id_clip", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    return args


def get_tree_tokens(args, steps):
    """
    Load the learned tokens into "prompt_to_vec" dict, there should be two new tokens per node per step.
    """
    prompt_to_vec = {}
    prompts_per_step = {}
    for step in steps:
        path_to_embed = f"{args.path_to_new_tokens}/{args.node}/{args.node}_seed{args.seed}/learned_embeds-steps-{step}.bin"
        assert os.path.exists(path_to_embed)
        data = torch.load(path_to_embed)
        prompts_per_step[step] = []
        combined = []
        for w_ in data.keys():
            key_ = w_.replace("<", "").replace(">","")
            new_key = f"<{key_}_{step}>" # <*_step> / <&_step>
            prompt_to_vec[new_key] = data[w_]
            combined.append(new_key)
            prompts_per_step[step].append(new_key)
        prompts_per_step[step].append(" ".join(combined))
    return prompt_to_vec, prompts_per_step


def plot_score(steps, score_list, name_list, save_name, args):
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    plt.title('Consistency Score')
    for score, name in zip(score_list, name_list):
        plt.plot(steps, score, label=name)
    plt.xlabel('Step')
    plt.ylabel('Consistency Score')
    plt.legend()
    plt.savefig(f"{args.path_to_new_tokens}/{args.node}/{args.node}_seed{args.seed}/consistency_test/{save_name}.png")



if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(f"{args.path_to_new_tokens}/{args.node}/{args.node}_seed{args.seed}/consistency_test"):
        os.mkdir(f"{args.path_to_new_tokens}/{args.node}/{args.node}_seed{args.seed}/consistency_test")
    
    prompts_title = ["Vl", "Vr", "Vl Vr"]
    steps = range(100, 1001, 100)
    prompt_to_vec, prompts_per_step = get_tree_tokens(args, steps)
    # prompts_per_step is {step: ["<*_step>", "<&_step>", "<*_step> <&_step>"]}
    print(prompts_per_step)
    
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to(args.device)
    utils.load_tokens(pipe, prompt_to_vec, args.device)

    print("Prompts loaded to pipe ...")
    print(prompt_to_vec.keys())

    clip_model = CLIPModel.from_pretrained(args.model_id_clip)
    preprocess = CLIPImageProcessor.from_pretrained(args.model_id_clip)

    prompts_to_images = {}
    prompts_to_clip_embeds = {}
    final_sim_score = {}
    gen_seeds = [4321, 95, 11, 87654]
    num_images_per_seed = 10
    for step in steps:
        print("=> Step: ", step)
        plt.figure(figsize=(20,15))
        prompts_to_images[step] = {}
        prompts_to_clip_embeds[step] = {}
        cur_prompts = prompts_per_step[step]
        for i in range(len(cur_prompts)):
            images_per_seed = []
            for gen_seed in gen_seeds:
                with torch.no_grad():
                    torch.manual_seed(gen_seed)
                    images = pipe(prompt=[cur_prompts[i]] * num_images_per_seed, num_inference_steps=25, guidance_scale=7.5).images
                images_per_seed.extend(images)
            
            # plot results
            plot_stacked = []
            for j in range(int(len(images_per_seed) / 4)):
                images_staked_h = np.hstack([np.asarray(img) for img in images_per_seed[j * 4:j * 4 + 4]])
                plot_stacked.append(images_staked_h)
            im_stack = np.vstack(plot_stacked)
            plt.subplot(1,len(cur_prompts) + 1, i + 1)
            plt.imshow(im_stack)
            plt.axis("off")
            plt.title(prompts_title[i], size=24)

            # saves the clip embeddings for all images
            images_preprocess = [preprocess(image, return_tensors="pt")["pixel_values"] for image in images_per_seed]
            stacked_images = torch.cat(images_preprocess)
            embedding_a = clip_model.get_image_features(stacked_images)
            emb_norm = torch.norm(embedding_a, dim=1)
            embedding_a = embedding_a / emb_norm.unsqueeze(1)

            prompts_to_images[step][cur_prompts[i]] = images_per_seed
            prompts_to_clip_embeds[step][cur_prompts[i]] = embedding_a

        # sim matrix per step
        cur_prompts = prompts_per_step[step]
        num_prompts = len(cur_prompts)
        sim_matrix = np.zeros((num_prompts, num_prompts))
        for i, k1 in enumerate(cur_prompts):
            for j, k2 in enumerate(cur_prompts):
                sim_mat = (prompts_to_clip_embeds[step][k1] @ prompts_to_clip_embeds[step][k2].T)
                if k1 == k2: 
                    sim_ = torch.triu(sim_mat, diagonal=1).sum() / torch.triu(torch.ones(sim_mat.shape), diagonal=1).sum()
                else:
                    sim_ = sim_mat.mean()
                sim_matrix[i, j] = sim_
        plt.subplot(1, len(cur_prompts) + 1, len(cur_prompts) + 1)
        plt.imshow(sim_matrix, vmin=0.4, vmax=0.9)
        for i, k1 in enumerate(prompts_title):
            plt.text(i, -0.9, f"{k1}", ha="center", va="center", size=16)
        for i, k2 in enumerate(prompts_title):
            plt.text(-1, i, f"{k2}", ha="center", va="center", size=16)
        for x in range(sim_matrix.shape[1]):
            for y in range(sim_matrix.shape[0]):
                plt.text(x, y, f"{sim_matrix[y, x]:.2f}", ha="center", va="center", size=18)
        plt.xlim([-1.5, len(cur_prompts) - 0.5])
        plt.ylim([len(cur_prompts)- 0.5, -1.5])
        plt.axis("off")
        
        
        s_l, s_r, s_lr = sim_matrix[0, 0], sim_matrix[1, 1], sim_matrix[0, 1]
        final_sim_score[step] = {}
        final_sim_score[step]['final'] = (s_l + s_r) + (min(s_l, s_r) - s_lr)
        final_sim_score[step]['s_l'] = s_l
        final_sim_score[step]['s_r'] = s_r
        final_sim_score[step]['s_lr'] = s_lr 
        plt.suptitle(f"Step Score [{final_sim_score[step]['final']:.2f}]", size=28)
        plt.savefig(f"{args.path_to_new_tokens}/{args.node}/{args.node}_seed{args.seed}/consistency_test/seed{args.seed}_step{step}.jpg")
    score_save_path = f"{args.path_to_new_tokens}/{args.node}/{args.node}_seed{args.seed}/consistency_test/seed{args.seed}_scores.bin"
    torch.save(final_sim_score, score_save_path)
    print(final_sim_score)
    
    # Load the score and plot
    final_sim_score = torch.load(score_save_path)
    final_score = []
    left_score = []
    right_score = []
    LR_score = []
    for i in steps:
        final_score.append(final_sim_score[i]['final'])
        left_score.append(final_sim_score[i]['s_l'])
        right_score.append(final_sim_score[i]['s_r'])
        LR_score.append(final_sim_score[i]['s_lr'])

    plot_score(steps, [final_score], ["Final Score"], "final_score", args)
    plot_score(steps, [left_score, right_score], ["Left Score", "Right Score"], "left_right_score", args)
    plot_score(steps, [LR_score], ["LR Score"], "lr_score", args)
