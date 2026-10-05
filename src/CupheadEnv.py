import math
import time

import cv2
import gymnasium as gym
import numpy as np

from Control import handle_action
from get_state import GameState

CHS = GameState()
ActionHandler = handle_action.HandleAction()


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
        self.min_delay = 0.05
        self.do_min_delay = True

    def step(self, action):
        # print(action)
        ActionHandler.ExecuteAction(action)
        ActionHandler.hold("x")

        #NEED TO FIX BECAUSE GPU FAST
        #can add a stopwatch set time between states(Start-stop wait how long that difference is between target fps)
        start_time = time.perf_counter()

        self.state = CHS.ReadYolo()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        if self.do_min_delay:
            time.sleep(self.min_delay - (elapsed_time) if elapsed_time < self.min_delay else 0)
            print(time.perf_counter() - start_time)

        # Calculate reward
        self.CurrentHP = self.state[4]
        done = False
        reward = 0.5

        if self.CurrentHP <= 0.0:
            ActionHandler.release("x")
            reward = -100.0
            done = True

        if not done:
            reward = -50.0 if self.CurrentHP < self.LastHP else reward
            reward -= 2.0 if not (0.15 < self.state[0] < 0.85) else 0
            reward += 12.5 if (self.lastXDel != 0.0 and (self.state[2] * self.lastXDel < 0) and self.state[3] < -0.2) else 0

            try:
                dist = math.hypot(self.state[2], self.state[3])
                if 0.2 <= dist <= 0.8:
                    reward += 2.0
                else:
                    dist_error = min(abs(dist - 0.2), abs(dist - 0.8))
                    reward -= dist_error * 10.0
            except Exception:
                pass

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
        ActionHandler.RetryLevel(self.CurrentHP)
        time.sleep(2.5)
        return CHS.ReadYolo()
