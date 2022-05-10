import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import yaml
import time
import pip
import sys
import soapy
import cv2

def loopFrame(sim, new_commands = None, full_action = None):
    t = time.time()
    sim.scrns = sim.atmos.moveScrns()
    sim.Tatmos = time.time() - t
    sim.dmCommands = new_commands
    sim.closed_correction = sim.runDM(
        sim.dmCommands, closed=True)
    # print(sim.closed_correction.shape)
    sim.slopes = sim.runWfs(dmShape=sim.closed_correction,
                            loopIter=sim.iters)
    # print(sim.slopes.shape)
    sim.open_correction = sim.runDM(sim.dmCommands,
                                    closed=False)
    if full_action is None:

        sim.combinedCorrection = sim.open_correction + sim.closed_correction
        print("///////////////////")
        print(sim.combinedCorrection.max())
        print(sim.combinedCorrection.dtype)
        print(sim.combinedCorrection.shape)
        print("////////////////////")
   # print(sim.open_correction.shape)
    else:
        sim.combinedCorrection = full_action
  #  print(sim.combinedCorrection.shape)
   # plt.imshow(sim.combinedCorrection[0])
   # plt.show()
    sim.runSciCams(sim.combinedCorrection)
    sim.storeData(sim.iters)
    sim.printOutput(sim.iters, strehl=True)
    sim.addToGuiQueue()
    sim.iters += 1
    return sim.combinedCorrection[0]

class AdaptiveOptics(gym.Env):
    def __init__(self, conf_file=None):
        if conf_file:
            self.conf_file = conf_file
        else:
            self.conf_file = sys.path[-1]+"/sh_8x8.yaml"
        with open(self.conf_file, 'r') as stream:
            self.data_loaded = yaml.safe_load(stream)
        self.__counter = 0
        self.last_reward = -1000
        self.reward = 0
        self.mem_img = []
        self.expert_commands = []
        self.action_space = spaces.Box(-100, 100, shape=(32,))
        self.observation_space = spaces.Box(-10000, 100000, shape=(64, 64, 1), dtype=np.uint8)
        self.pre_expert_value = None
        self.expert_value = None
        self.max_reward = 5
        self.min_reward = 0
        self.mean_reward = 0
        self._initao()

    def _initao(self):
        #self.data_loaded['Atmosphere']['windDirs'] = np.random.randint(0, 180, 4).tolist()
        self.sim = soapy.Sim(self.conf_file)
        self.sim.aoinit()
        self.sim.makeIMat()

        self.mem_img = []
        for i in range(3):
            expert_value = self.expert()
            loopFrame(self.sim, expert_value)
            img = self.sim.sciImgs[0].copy()
         #   img = ((img - np.min(img)) / (np.max(img) - np.min(img))) * 255
            img = img.astype(np.uint8)
            img = img.reshape(1, 64, 64)
            self.mem_img.append(img)
        self.pre_expert_value = self.expert()

        return img

    def next_phase(self, action):
        expert_do = loopFrame(self.sim, action)
        img = self.sim.sciImgs[0].copy()
        return np.asarray((img, cv2.resize(expert_do, (64,64))))

    def expert(self):
        if self.sim.config.sim.nDM:
            self.sim.dmCommands[:] = self.sim.recon.reconstruct(self.sim.slopes)
        commands = self.sim.buffer.delay(self.sim.dmCommands, self.sim.config.sim.loopDelay)
        return commands

    def step(self, action):
        phase = loopFrame(self.sim, self.expert(), cv2.resize(action, (140, 140)).reshape(1,140,140))
       # phase = loopFrame(self.sim, self.action_space.sample())
        img = self.sim.sciImgs[0].copy()
        self.reward = np.sum(img**2) / np.sum(img)**2
        done = False
        return [img, cv2.resize(phase, (64, 64))], self.reward*100, False, {}

    def reset(self):
        state = self._initao()
        return state

    def render(self):
        plt.imshow(self.sim.sciImgs[0])
        plt.show()

    def close(self):
        pass

