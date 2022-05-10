from gym import Wrapper
from gym import spaces, logger
import numpy as np

class FrameStack(Wrapper):
    def __init__(self, env, stack = 20):
        super(FrameStack, self).__init__(env)
        self.stack = 20
        self.observation_space = spaces.Box(-10000, 100000, shape=(self.stack,2, 64, 64), dtype=np.uint8)

    def reset(self):
        self.obs_stack = [None]*self.stack
        for i in range(self.stack*5):
            obs = np.asarray(self.env.next_phase(self.env.expert()))
          #  print(len(obs))
            self.obs_stack.append(obs)
            self.obs_stack[:self.stack] = self.obs_stack[1:self.stack+1]
            popv = self.obs_stack.pop(-1)

        obs_stack = np.array(self.obs_stack)
        print(obs_stack.shape)

        return obs_stack.reshape( self.stack,2, 64,64)

    def step(self , action):
        obs, reward, done, info = super().step(action)
      #  print(obs[0].shape)
       # print(obs[1].shape)
        self.obs_stack.append(tuple(obs))
        self.obs_stack[:self.stack] = self.obs_stack[1:self.stack+1]
        self.obs_stack.pop(-1)
        return np.asarray(self.obs_stack).reshape(self.stack, 2, 64,64), reward, done, info
