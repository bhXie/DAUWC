import numpy as np
import torch
import gym
import argparse
import os
import d4rl
from tqdm import trange
from coolname import generate_slug
import time
import json
from mylog import Logger

import utils
import DAUWC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment

# set path 
d4rl.set_dataset_path('/data/dataset')


def eval_policy(args, iter, logger: Logger, policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    lengths = []
    returns = []
    avg_reward = 0.
    for _ in range(eval_episodes):
        
        state, done = eval_env.reset(), False
        steps = 0
        episode_return = 0
        while not done:
            state = (np.array(state).reshape(1, -1) - mean)/std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            episode_return += reward
            steps += 1
        lengths.append(steps)
        returns.append(episode_return)

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward)*100

    logger.log('eval/lengths_mean', np.mean(lengths), iter)
    logger.log('eval/lengths_std', np.std(lengths), iter)
    logger.log('eval/returns_mean', np.mean(returns), iter)
    logger.log('eval/returns_std', np.std(returns), iter)
    logger.log('eval/d4rl_score', d4rl_score, iter)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {d4rl_score:.3f}")
    print("---------------------------------------")
    return d4rl_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="DAUWC")              
    parser.add_argument("--env", default="halfcheetah-medium-v2")        
    parser.add_argument("--seed", default=0, type=int)              
    parser.add_argument("--eval_freq", default=1e4, type=int)      
    parser.add_argument("--max_timesteps", default=1e6, type=int)   
    parser.add_argument("--save_model", action="store_true", default=False)       
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument("--normalize", default=False, action='store_true')
    parser.add_argument("--batch_size", default=256, type=int)     
    parser.add_argument("--temperature", default=3.0, type=float)
    parser.add_argument("--v", default=0.3, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--discount", default=0.99, type=float)    
    parser.add_argument('--work_dir', default='tmp', type=str)
    args = parser.parse_args()
    args.cooldir = generate_slug(2)

    # Build work dir
    base_dir = 'runs'
    utils.make_dir(base_dir)
    base_dir = os.path.join(base_dir, args.work_dir)
    utils.make_dir(base_dir)
    args.work_dir = os.path.join(base_dir, args.env)
    utils.make_dir(args.work_dir)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d-%H:%M", ts)
    exp_name = 'test-' + str(args.env) + '-' + ts + '-bs' + str(args.batch_size) + '-s' + str(args.seed)
    if args.policy == 'DAUWC':
        exp_name += '-t' + str(args.temperature) + '-e' + str(args.v)
    else:
        raise NotImplementedError
    exp_name += '-' + args.cooldir
    args.work_dir = args.work_dir + '/' + exp_name 
    utils.make_dir(args.work_dir)

    args.model_dir = os.path.join(args.work_dir, 'model')
    utils.make_dir(args.model_dir)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    utils.snapshot_src('.', os.path.join(args.work_dir, 'src'), '.gitignore')

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        # DAUWC
        "discount": args.discount,
        "tau": args.tau,
        "temperature": args.temperature,
        "v": args.v,
    }

    # Initialize policy
    if args.policy == 'DAUWC':
        policy = DAUWC.DAUWC(**kwargs)
    else:
        raise NotImplementedError

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    if 'antmaze' in args.env:
        # Center reward for Ant-Maze
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        replay_buffer.reward = replay_buffer.reward - 1.0
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    logger = Logger(args.work_dir, use_tb=True)

    for t in trange(int(args.max_timesteps)):
        policy.train(replay_buffer, args.batch_size, logger=logger)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            eval_episodes = 100 if t+1 == int(args.max_timesteps) else args.eval_episodes
            d4rl_score = eval_policy(args, t+1, logger, policy, args.env,
                                     args.seed, mean, std, eval_episodes=eval_episodes)
            if args.save_model:
                policy.save(args.model_dir)

    logger._sw.close()
