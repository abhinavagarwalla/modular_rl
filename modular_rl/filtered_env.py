from gym import Env, spaces
import numpy as np
import opensim as osim
from osim.env import *

class FilteredEnv(RunEnv): #pylint: disable=W0223
    def __init__(self, env, ob_filter=None, rew_filter=None):
        self.env = env
        self.ob_filter = ob_filter
        self.rew_filter = rew_filter

        ob_space = self.env.observation_space
        shape = self.ob_filter.output_shape(ob_space)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape)
        self.action_space = self.env.action_space

    def _step(self, ac):
        ob, rew, done, info = self.env.step(ac)
        nob = self.ob_filter(ob) if self.ob_filter else ob
        nrew = self.rew_filter(rew) if self.rew_filter else rew
        info["reward_raw"] = rew
        return (nob, nrew, done, info)

    def _reset(self, difficulty=2, seed=None):
        ob = self.env.reset(difficulty, seed)
        return self.ob_filter(ob) if self.ob_filter else ob

    def _render(self, *args, **kw):
        self.env.render(*args, **kw)
