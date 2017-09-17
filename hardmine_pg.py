#!/usr/bin/env python
"""
This script runs a policy gradient algorithm
"""

from gym.envs import make
from modular_rl import *
import argparse, sys, cPickle
from tabulate import tabulate
import shutil, os, logging
import gym
import opensim as osim
from osim.env import *
from multiprocessing import Pool

def animate_rollout_save(agent, seed, iffilter, n_timesteps=1000, delay=.01):
    total_reward = 0.
    env = RunEnv(False)
    ob = env.reset(seed=seed)
    if iffilter==2:
        ofd = FeatureInducer(env.observation_space)
    elif iffilter==1:
        ofd = ConcatPrevious(env.observation_space)
    for i in range(n_timesteps):
        ob = ofd(ob)
        ob = agent.obfilt(ob)
        a, _info = agent.act(ob)
        ob, _rew, done, _info = env.step(a)
        total_reward += _rew
        ob = np.array(ob)
        if done:
            print(("terminated after %s timesteps"%i))
            break
        time.sleep(delay)
    print("Reward={}, Seed={}, Timesteps={}".format(total_reward, seed, i))

def parallel_animate((agent, seed, iffilter)):
    animate_rollout_save(agent, seed, iffilter, 1000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)
    parser.add_argument("--env",required=True)
    parser.add_argument("--agent",required=True)
    parser.add_argument("--plot",action="store_true")
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])
    
    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    args = parser.parse_args()
    if args.timestep_limit == 0:
        args.timestep_limit = 1000 #env_spec.timestep_limit
    np.random.seed(args.seed)

    if args.use_hdf:
        if args.load_snapshot:
            hdf = load_h5_file(args)
            key = hdf["agent_snapshots"].keys()[-1]
            latest_snapshot = hdf["agent_snapshots"][key]
            agent = cPickle.loads(latest_snapshot.value)
            agent.stochastic=False

            args_list = [(agent,
                  seed,
                  args.filter
                  ) for seed in np.random.randint(1, 5000, 1000)]

            p = Pool(args.parallel)
            p.map(parallel_animate, args_list)
            # parallel_animate(args_list[0])