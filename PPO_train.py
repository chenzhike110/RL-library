import argparse
import gym
import torch
import time
import torch.multiprocessing as mp
mp.set_start_method('spawn', True)
from torch.utils.tensorboard import SummaryWriter
tb_writer = SummaryWriter()

from PPO.PPO import PPO
from PPO.memory import Memory
from PPO.agent import Agent

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.99,
                    help='discount factor')
    parser.add_argument("--tau", type=float, default=0.97,
                    help='gae')
    parser.add_argument("--env-name", type=str, default="Humanoid-v2",
                    help="env name")
    parser.add_argument("--max-timestep", type=int, default=50000,
                    help="max timestep every episode")
    parser.add_argument("--max-episode", type=int, default=800000,
                    help="max episode time")
    parser.add_argument("--log_interval", type=int, default=200,
                    help="interval to update log")
    parser.add_argument("--update-timestep", type=int, default=800,
                    help="update time step")
    parser.add_argument("--n-latent-var", type=int, default=64,
                    help="hidden layer number")
    parser.add_argument("--eps-clip", type=float, default=0.2,
                    help="clip boundary")
    parser.add_argument("--k-epoches", type=int, default=4,
                    help="network update frequency in a eposide")
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999),
                    help="optimization beta")
    parser.add_argument("--lr", type=float, default=0.002,
                    help="learning rate")
    parser.add_argument("--num-agents", type=int, default=2,
                    help="agent number for parallel")   
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    args = parser.parse_args()
    return args

def main():
    args = parse_arg()
    sample_env = gym.make(args.env_name)
    ppo = PPO(sample_env, args)
    memory = Memory(sample_env, ppo.policy_old, args)

    agents = []
    pipes = []

    update_request = [False]*args.num_agents
    agent_completed = [False]*args.num_agents

    update_iteration = 0
    log_iteration = 0
    average_eps_reward = 0
    reward_record = [[None]*args.num_agents]

    for agent_id in range(args.num_agents):
        p_start, p_end = mp.Pipe()
        agent = Agent(str(agent_id), args.env_name, memory, p_end, args)

        agent.start()
        agents.append(agent)
        pipes.append(p_start)
    while True:
        for i, conn in enumerate(pipes):
            if conn.poll():
                msg = conn.recv()

                if type(msg).__name__ == "MsgMaxReached":
                    agent_completed[i] = True
                # if agent is waiting for network update
                elif type(msg).__name__ == "MsgUpdateRequest":
                    update_request[i] = True
                    if False not in update_request:
                        ppo.update(memory)
                        update_iteration += 1
                        update_request = [False]*args.num_agents
                        msg = update_iteration
                        # send to signal subprocesses to continue
                        for pipe in pipes:
                            pipe.send(msg)
                elif type(msg).__name__ == "MsgRewardInfo":
                    idx = int(msg.episode/args.log_interval)
                    if len(reward_record) <= idx:
                        reward_record.append([None]*args.num_agents)
                    reward_record[idx-1][i] = msg.reward

                    # if all agents has sent msg for this logging iteration
                    if (None not in reward_record[log_iteration]):
                        eps_reward = reward_record[log_iteration]
                        average_eps_reward = 0
                        for i in range(len(eps_reward)):
                            print("Agent {} Episode {}, Avg Reward/Episode {:.2f}"
                                  .format(i, (log_iteration+1)*args.log_interval,
                                          eps_reward[i]))
                            average_eps_reward += eps_reward[i]

                            tb_writer.add_scalar("Agent_{}_Episodic_Reward".format(i), eps_reward[i], (log_iteration+1)*args.log_interval, time.time())
                        print("Main: Update Iteration: {}, Avg Reward Amongst Agents: {:.2f}\n"
                              .format(update_iteration,
                                      average_eps_reward/args.num_agents))
                        tb_writer.add_scalar("Avg_Agent_reward", average_eps_reward/args.num_agents, update_iteration, time.time())
                        log_iteration += 1
        if False not in agent_completed:
            print("=Training ended with Max Episodes=")
            break
    
    for agent in agents:
        agent.terminate()

if __name__ == "__main__":
    main()
