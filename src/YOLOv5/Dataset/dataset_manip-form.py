import time
import os
import glob
from window_capture import WindowCapture

output_dir = "saved_dataset"
os.makedirs(output_dir, exist_ok=True)

def gen_data():
    wc = WindowCapture("Cuphead")
    i = 0
    n = input("Max Number: ")
    t = input("Time Between: ")
    try:
        while True and i<=n:
            filename = os.path.join(output_dir, f"screenshot_{i:03}.png")
            wc.save_screenshot(filename)
            print(f"Saved {filename}")
            i += 1
            time.sleep(t)
    except KeyboardInterrupt:
        print("\nCapturing Interrupted")

def format_data():
    files = sorted(glob.glob(os.path.join(output_dir, "screenshot_*.png")))

    for new_index, old_path in enumerate(files):
        new_filename = f"screenshot_{new_index:03}.png"
        new_path = os.path.join(output_dir, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} -> {new_path}")

select = int(input("Select Manipulation:    Generate Data[0]    Format Data[1]\n"))

if select == 0:
    gen_data()
    input("\nFinished Data Generation")
elif select == 1:
    format_data()
    input("\nRenumbering complete!")
else:
    input("Invalid Selection")
