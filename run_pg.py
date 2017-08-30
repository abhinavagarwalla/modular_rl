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
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])
    
    env = RunEnv(False)
    # env = make(args.env)
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
    cfg = args.__dict__
    np.random.seed(args.seed)

    if args.filter:
        ofd = ConcatPrevious(env.observation_space)
        env = FilteredEnv(env, ob_filter=ofd)

    agent = agent_ctor(env.observation_space, env.action_space, cfg)
    print("Agent actor has been loaded")

    if args.use_hdf:
        if args.load_snapshot:
            hdf = load_h5_file(args)
            key = hdf["agent_snapshots"].keys()[-1]
            latest_snapshot = hdf["agent_snapshots"][key]
            agent = cPickle.loads(latest_snapshot.value)
            COUNTER = int(key)
        else:
            hdf, diagnostics = prepare_h5_file(args)
            COUNTER = 0
    gym.logger.setLevel(logging.WARN)

    def callback(stats):
        global COUNTER
        COUNTER += 1
        # Print stats
        print("*********** Iteration %i ****************" % COUNTER)
        print(tabulate([k_v for k_v in list(stats.items()) if np.asarray(k_v[1]).size==1])) #pylint: disable=W0110
        # Store to hdf5
        if args.use_hdf:
            if not args.load_snapshot:
                for (stat,val) in list(stats.items()):
                    if np.asarray(val).ndim==0:
                        diagnostics[stat].append(val)
                    else:
                        assert val.ndim == 1
                        diagnostics[stat].extend(val)
            if args.snapshot_every and ((COUNTER % args.snapshot_every==0) or (COUNTER==args.n_iter)):
                hdf['/agent_snapshots/%0.4i'%COUNTER] = np.array(cPickle.dumps(agent,-1))
        # Plot
        if args.plot:
            animate_rollout(env, agent, min(1000, args.timestep_limit))

    run_policy_gradient_algorithm(env, agent, callback=callback, usercfg = cfg)

    if args.use_hdf:
        try:
            hdf['env_id'] = env_spec.id 
            hdf['env'] = np.array(cPickle.dumps(env, -1))
        except Exception: print("failed to cPickle env") #pylint: disable=W0703
    env.close()
