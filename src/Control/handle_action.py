import time
from enum import Enum

from .handle_keypress import HandleKeypress

DISABLE_FAILSAFE_INTERVAL = True


class Action(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    SHOOT = "x"
    JUMP = "z"
    DASH = "shift"


class Release(str, Enum):
    UP = "up"
    LEFT = "left"
    RIGHT = "right"
    JUMP = "z"
    DASH = "shift"


class HandleAction:
    @staticmethod
    def RetryLevel(health):
        HandleAction.release_all()
        if health == 0:
            time.sleep(2)
            HandleAction.execute("enter")
            time.sleep(1.25)
        elif health == -1:
            time.sleep(10)
            HandleAction.execute("enter")
            HandleAction.WorldStartLevel()
        pass

    @staticmethod
    def WorldStartLevel():
        time.sleep(4)
        HandleAction.hold("up")
        time.sleep(0.3)
        HandleAction.release("up")
        HandleAction.execute("z")
        time.sleep(1)
        HandleAction.execute("z")
        time.sleep(3)

    @staticmethod
    def ExecuteAction(action):
        direction = action % 2
        crouch = action // 4
        dash = (action // 2) % 2

        if direction == 0:
            HandleAction.hold("left")
        elif direction == 1:
            HandleAction.hold("right")

        if crouch == 1:
            HandleAction.hold("down")
        else:
            HandleAction.release("down")

        if dash == 1:
            HandleAction.execute("shift")
        time.sleep(0.025)
        # print(f"Direction:{direction} Crouch:{crouch} Dash:{dash}")
        HandleAction.release_select()

    @staticmethod
    def execute(action, interval=0.1):
        HandleKeypress.press(action, interval)

    @staticmethod
    def hold(action):
        HandleKeypress.hold(action)

    @staticmethod
    def release(action):
        HandleKeypress.release(action)

    @staticmethod
    def release_all():
        HandleKeypress.release([a.value for a in Action])

    @staticmethod
    def release_select():
        HandleKeypress.release([a.value for a in Release])
