#import pymem.process
import numpy as np
import pymem
import pymem.process


class pointer_reader:
    def __init__(self, ProcessName = "Cuphead.exe", ModuleName = "mono.dll", PointerOffsets = [0x264A68, 0xA0, 0xD20, 0xE8, 0x20, 0xB4]):
        self.ProcessName = ProcessName
        self.ModuleName = ModuleName
        self.PointerOffsets = PointerOffsets

    def get_health(self, normalize = False):
        try:
            pm = pymem.Pymem(self.ProcessName)
        except pymem.exception.PymemError:
            print("Game is not detected.")
            return -1.0

        try:
            address = pymem.process.module_from_name(pm.process_handle, self.ModuleName).lpBaseOfDll

            for offset in self.PointerOffsets[:-1]:
                address = pm.read_longlong(address + offset)

            address = address + self.PointerOffsets[-1]
            read_health = pm.read_int(address)
            pm.close_process()

            if read_health < 0:
                return -1
            else:
                if normalize:
                    return np.clip(read_health / 3, 0.0, 1.0)
                else:
                    return read_health
        except Exception:
            return -1.0
