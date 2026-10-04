import time

import numpy as np

from State.pointer_reader import pointer_reader
from YOLOv5.WindowCapture.window_capture import WindowCapture
from YOLOv5.yolo import YoloDetection


class CHState:
    def __init__(
        self, WindowName="CVCuphead", IconPath="CupheadAIProject.ico", Debug=False
    ):
        self.yolo = YoloDetection()
        self.pointer_reader = pointer_reader()
        self.window_capture = WindowCapture("Cuphead")
        time.sleep(1)
        self.WindowName = WindowName
        self.IconPath = IconPath
        self.Debug = Debug

    def ReadYolo(self):
        self.screenshot = self.window_capture.get_screenshot()
        self.prediction = self.yolo.detect_single_image(self.screenshot)
        self.ConfFiltVector = self.yolo.process_objects()
        self.ConfFiltVector = np.append(
            self.ConfFiltVector, self.pointer_reader.get_health(normalize=True)
        )
        if self.Debug:
            print(self.ConfFiltVector)
        return self.ConfFiltVector

    def ShowYolo(self):
        self.prediction.show(1, self.WindowName)
        #hwnd = win32gui.FindWindow(None, self.WindowName)
        """
        Some error with the LoadImage part
        win32gui.SendMessage(hwnd, win32con.WM_SETICON, win32con.ICON_BIG, win32gui.LoadImage(None, self.IconPath, win32con.IMAGE_ICON, 0, 0, win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE))
        """

def main():
    yolo = YoloDetection()
    pointer_reader = pointer_reader()
    window_capture = WindowCapture("Cuphead")
    time.sleep(1)
    WindowName = "CVCuphead"
    while True:
        screenshot = window_capture.get_screenshot()
        prediction = yolo.detect_single_image(screenshot)
        ConfFiltVector = yolo.process_objects()
        print(np.append(ConfFiltVector, pointer_reader.get_health()))
        prediction.show(1, WindowName)
        #hwnd = win32gui.FindWindow(None, WindowName)
        #icon_path = "CupheadAIProject.ico"
        #win32gui.SendMessage(hwnd, win32con.WM_SETICON, win32con.ICON_BIG, win32gui.LoadImage(None, IconPath, win32con.IMAGE_ICON, 0, 0, win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE))

if __name__ == "__main__":
    main()
