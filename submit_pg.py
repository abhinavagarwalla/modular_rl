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
from osim.http.client import Client

remote_base = 'http://grader.crowdai.org:1729'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)
    parser.add_argument("--env",required=True)
    parser.add_argument("--agent",required=True)
    parser.add_argument('--token', dest='token', action='store', required=True)
    # parser.add_argument("--filteryn", default=False, action="store_true")
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])
    
    env = RunEnv(False)
    # env = make(args.env)
    env_spec = env.spec
    client = Client(remote_base)

    # Create environment
    observation = client.env_create(args.token)
        
    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    args = parser.parse_args()
    if args.timestep_limit == 0:
        args.timestep_limit = env_spec.timestep_limit
    cfg = args.__dict__
    np.random.seed(args.seed)

    agent = agent_ctor(env.observation_space, env.action_space, cfg)
    hdf = load_h5_file(args)
    key = hdf["agent_snapshots"].keys()[-2]
    latest_snapshot = hdf["agent_snapshots"][key]    
    agent = cPickle.loads(latest_snapshot.value)

    while True:
        ob = agent.obfilt(observation)
        a, _info = agent.act(ob)
        [observation, reward, done, info] = client.env_step(a.tolist())
        # print(observation)
        if done:
            observation = client.env_reset()
            latest_snapshot = hdf["agent_snapshots"][key]    
            agent = cPickle.loads(latest_snapshot.value)
            if not observation:
                break

    client.submit()
