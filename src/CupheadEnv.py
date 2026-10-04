import math
import time

import cv2
import gymnasium as gym
import numpy as np

from Control import handle_action
from get_state import GameState

CHS = GameState()


class CupheadEnv(gym.Env):
    def __init__(self):
        self.LastHP = 1.0
        self.CurrentHP = 1.0
        # Direction(Left/Right) Crouch(False/True) Dash(False/True)
        self.action_space = gym.spaces.Discrete(8)
        # self.action_space = gym.spaces.MultiDiscrete([2, 2, 2])
        # 5 long array with values -1 to 1 with the data type np.float32
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )
        self.lastXDel = 0.0

    def step(self, action):
        handle_action.HandleAction.ExecuteAction(action)
        # print(action)
        handle_action.HandleAction.hold("x")

        #NEED TO FIX BECAUSE GPU FAST
        #can add a stopwatch set time between states(Start-stop wait how long that difference is between target fps)
        time.sleep(.1)
        self.state = CHS.ReadYolo()

        # Calculate reward
        self.CurrentHP = self.state[4]
        done = False
        reward = 0.5

        if self.CurrentHP <= 0.0:
            handle_action.HandleAction.release("x")
            reward = -100.0
            done = True

        if not done:
            if self.CurrentHP < self.LastHP:
                reward = -50.0
            try:
                dist = math.hypot(self.state[2], self.state[3])

                if 0.2 <= dist <= 0.8:
                    reward += 2.0
                else:
                    dist_error = min(abs(dist - 0.2), abs(dist - 0.8))
                    reward -= dist_error * 10.0
            except error as e:
                pass
            if not (0.15 < self.state[0] < 0.85):
                reward -= 2.0

            if (
                self.lastXDel != 0.0
                and (self.state[2] * self.lastXDel < 0)
                and self.state[3] < -0.2
            ):
                reward += 12.5
        print(reward)
        self.lastXDel = self.state[2]
        # Render Visuals
        CHS.ShowYolo()
        # time.sleep(10000)
        self.LastHP = self.CurrentHP
        # Set placeholder for info
        info = {}
        # Return step information
        print(self.state)
        return self.state, reward, done, info

    def render(self, mode):
        # CHS.ShowYolo()
        pass

    def reset(self):
        self.LastHP = 1.0
        self.lastXDel = 0.0
        cv2.destroyAllWindows()
        handle_action.HandleAction.RetryLevel(self.CurrentHP)
        time.sleep(2.5)
        return CHS.ReadYolo()
