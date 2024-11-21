import torch
import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", type=str, help="Path to directory with the new embeddings")
    parser.add_argument("--node1", type=str, help="Node name")
    parser.add_argument("--seed1", type=int, default=111)
    parser.add_argument("--name1", type=str, default='Trial 1')
    parser.add_argument("--path2", type=str, help="Path to directory with the new embeddings")
    parser.add_argument("--node2", type=str, help="Node name")
    parser.add_argument("--seed2", type=int, default=111)
    parser.add_argument("--name2", type=str, default='Trial 2')

    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    return args


def plot(args):
    conScore1 = []
    conScore2 = []
    try:
        score1 = torch.load(f'{args.path1}/{args.node1}/{args.node1}_seed{args.seed1}/consistency_test/seed{args.seed1}_scores.bin')
        score2 = torch.load(f'{args.path2}/{args.node2}/{args.node2}_seed{args.seed2}/consistency_test/seed{args.seed2}_scores.bin')
    except:
        raise Exception("No consistency score found")
    
    min_step = min(min(score1.keys()), min(score2.keys()))
    max_step = max(max(score1.keys()), max(score2.keys()))
    step = 100
    for i in range(100, 1001, 100):
        conScore1.append(score1[i]['final'])
        conScore2.append(score2[i]['final'])

    plt.figure(figsize=(6.4, 4.8))
    plt.title('Consistency Score')
    plt.plot(range(min_step, max_step + 1, step), conScore1, label=args.name1)
    plt.plot(range(min_step, max_step + 1, step), conScore2, label=args.name2)
    plt.xlabel('Step')
    plt.ylabel('Consistency Score')
    plt.legend()
    plt.savefig(f'{args.path1}/comparison_{args.name1}_{args.name2}.png')
    

if __name__ == '__main__':
    args = parse_args()
    plot(args)