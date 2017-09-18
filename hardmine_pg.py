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

def read_seeds_file(filename='seed_res.txt'):
    f = map(str.strip, open(filename).readlines())
    seed_dict = {'Reward': [], 'Seed': []}
    for i in f:
        seed_dict['Reward'].append(float(i.split(',')[0].split('=')[-1]))
        seed_dict['Seed'].append(int(i.split(',')[1].split('=')[-1]))
    print(seed_dict)
    return seed_dict

def make_env(args):
    env = RunEnv(False)
    env_spec = env.spec

    mondir = args.outfile + ".dir"
    if args.load_snapshot:
        env = gym.wrappers.Monitor(env, mondir, video_callable=None if args.video else VIDEO_NEVER, resume=True)
    else:
        if os.path.exists(mondir): shutil.rmtree(mondir)
        os.mkdir(mondir)
        env = gym.wrappers.Monitor(env, mondir, video_callable=None if args.video else VIDEO_NEVER)
    
    if args.filter==2:
        ofd = FeatureInducer(env.observation_space)
        env = FilteredEnv(env, ob_filter=ofd)
    elif args.filter==1:
        ofd = ConcatPrevious(env.observation_space)
        env = FilteredEnv(env, ob_filter=ofd)
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)
    parser.add_argument("--env",required=True)
    parser.add_argument("--agent",required=True)
    parser.add_argument("--plot",action="store_true")
    parser.add_argument("--train", action="store_true")
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])
    
    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    args = parser.parse_args()
    if args.timestep_limit == 0:
        args.timestep_limit = 1000 #env_spec.timestep_limit
    # np.random.seed(args.seed)

    if args.use_hdf:
        if args.load_snapshot:
            hdf = load_h5_file(args)
            key = hdf["agent_snapshots"].keys()[-1]
            latest_snapshot = hdf["agent_snapshots"][key]
            agent = cPickle.loads(latest_snapshot.value)

            if not args.train:
                agent.stochastic=False
                args_list = [(agent,
                      seed,
                      args.filter
                      ) for seed in np.random.randint(1, 5000, 3000)]

                # p = Pool(args.parallel)
                # p.map(parallel_animate, args_list)
                for i in args_list:
                    parallel_animate(i)
            elif args.train:
                seed_dict = read_seeds_file()
                seed_iter = get_seed_iter(seed_dict)
                COUNTER = int(key)

                env = make_env(args)
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
                            print("Saving model at COUNTER=", COUNTER)
                            hdf['/agent_snapshots/%0.4i'%COUNTER] = np.array(cPickle.dumps(agent,-1))
                            print("Model saving done")
                    # Plot
                    if args.plot:
                        animate_rollout(env, agent, min(1000, args.timestep_limit))

                run_policy_gradient_algorithm_hardmining(env, agent, callback=callback, usercfg = cfg)
                env.close()