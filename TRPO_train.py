import torch
import gym
import argparse
from TRPO.TRPO import agent

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.99,
                    help='discount factor')
    parser.add_argument("--tau", type=float, default=0.97,
                    help='gae')
    parser.add_argument("--batch-size", type=int, default=15000,
                    help='training batch size')
    parser.add_argument("--l2-reg", type=float, default=1e-3,
                    help="l2 regularization regression")
    parser.add_argument("--max-kl",type=float, default=1e-2,
                    help="max kl value")
    parser.add_argument("--damping", type=float, default=0.1,
                    help="dampling")
    parser.add_argument('--seed', type=int, default=543,
                    help='random seed (default: 1)')
    parser.add_argument("--nsteps", type=int, default=10,
                    help="search step")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arg()
    env = gym.make("Humanoid-v2")
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    model = agent(env, args)
    model.learn()
