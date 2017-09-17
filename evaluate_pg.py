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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)
    parser.add_argument("--env",required=True)
    parser.add_argument("--agent",required=True)
    parser.add_argument("--plot",action="store_true")
    # parser.add_argument("--filteryn", default=False, action="store_true")
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])
    
    env = RunEnv(False)
    env_spec = env.spec
    
    mondir = args.outfile + ".dir"
    if args.load_snapshot:
        env = gym.wrappers.Monitor(env, mondir, video_callable=None if args.video else VIDEO_NEVER, resume=True)
    else:
        if os.path.exists(mondir): shutil.rmtree(mondir)
        os.mkdir(mondir)
        env = gym.wrappers.Monitor(env, mondir, video_callable=None if args.video else VIDEO_NEVER)
    
    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    args = parser.parse_args()
    if args.timestep_limit == 0:
        args.timestep_limit = env_spec.timestep_limit
    np.random.seed(args.seed)

    if args.use_hdf:
        if args.load_snapshot:
            hdf = load_h5_file(args)
            print(hdf["agent_snapshots"].keys()[-2])
            for key in hdf["agent_snapshots"].keys()[-1:]:
                for i in range(3):
                    print("Evaluating with key: ", key, " , i=", i)
                    latest_snapshot = hdf["agent_snapshots"][key]
                    agent = cPickle.loads(latest_snapshot.value)
                    agent.stochastic=False
                    animate_rollout(env, agent, min(1000, args.timestep_limit))
