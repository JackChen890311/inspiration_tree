import os
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help="Path to output directory", default="outputs/")
    parser.add_argument("--node_name", type=str, help="Node name", default="v0")
    parser.add_argument("--exp_file_name", type=str, help="Exp file name", default="outputs/exp.txt")
    args = parser.parse_args()
    return args


def collect_scores(args):
    print("Collecting scores from", args.output_path, args.node_name)
    scores_dict = {}

    # Collect scores from output_path
    for concept in os.listdir(args.output_path):
        if concept == args.exp_file_name.split("/")[-1]:
            continue
        for node_seed in os.listdir(f"{args.output_path}/{concept}/{args.node_name}"):
            node, seed = node_seed.split("_")
            seed = int(seed[4:])
            path = f"{args.output_path}/{concept}/{args.node_name}/{node_seed}/consistency_test/seed{seed}_scores.bin"
            if os.path.exists(path):
                all_scores = torch.load(path)
            scores_dict[concept] = round(all_scores[1000]['final'], 4)
    return scores_dict


def write_to_exp_file(scores_dict, args):
    with open(args.exp_file_name, "a") as f:
        for concept, score in scores_dict.items():
            f.write(f"{concept}: {score}\n")
        average_score = round(sum(scores_dict.values()) / len(scores_dict), 4) if len(scores_dict) > 0 else -1
        f.write(f"Average: {average_score}\n")

if __name__ == "__main__":
    args = parse_args()
    print(args)
    scores_dict = collect_scores(args)
    print(scores_dict)
    write_to_exp_file(scores_dict, args)

