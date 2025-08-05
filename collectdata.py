import argparse, json, gzip, queue, threading, time, os
from pathlib import Path
from datetime import datetime

import numpy as np, pandas as pd, h5py, websocket
import tkinter as tk
import yaml

IP = None
PORT = None
SAMPLE_RATE = None
WINDOW_MS = None
STRIDE_MS = None
T = None 

GESTURE_PLAN = []

HEX_PER_TAXEL   = 3
TAXELS_EXPECTED = 16

def load_gesture_plan(yaml_file: str):
    """Load gesture plan from YAML file."""
    global GESTURE_PLAN
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        GESTURE_PLAN = [(g['name'], g['duration']) for g in data['gestures']]
        print(f"[INFO] Loaded {len(GESTURE_PLAN)} gestures from {yaml_file}")
    except FileNotFoundError:
        print(f"[ERROR] Gesture file {yaml_file} not found.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Invalid YAML in {yaml_file}: {e}")
        exit(1)
    except KeyError as e:
        print(f"[ERROR] Missing required key in YAML: {e}")
        exit(1)

class PoseGUI:
    """Tk window that shows pose name and countdown (no GIF)."""
    def __init__(self, window_px: int = 500):
        self.root = tk.Tk()
        self.root.title("Data-collection instructions")
        self.root.geometry(f"{window_px}x{window_px}")
        self.root.resizable(False, False)

        spacer = tk.Frame(self.root, height=window_px // 4)
        spacer.pack()

        self.lbl_pose = tk.Label(self.root, font=("Helvetica", 28, "bold"))
        self.lbl_pose.pack(pady=6)

        self.lbl_time = tk.Label(self.root, font=("Helvetica", 56))
        self.lbl_time.pack(pady=10)

    def show_prepare(self, pose: str):
        self._switch_pose(pose)
        self.lbl_time.config(text="3")

    def show_perform(self, pose: str):
        self._switch_pose(pose)

    def update_seconds(self, sec_left: int):
        self.lbl_time.config(text=str(sec_left))

    def update(self):
        """Process pending Tk events (non-blocking)."""
        self.root.update_idletasks()
        self.root.update()

    def _switch_pose(self, pose: str):
        self.lbl_pose.config(text=pose.upper())

SENSOR_ID  = None
last_msg   = {}
raw_q      = queue.Queue()

def on_message(wsapp, message):
    global SENSOR_ID, last_msg
    try:
        data = json.loads(message)
        if data.get("type") == "welcome":
            return
        last_msg = data
        if SENSOR_ID is None:
            SENSOR_ID = next((k for k in data if k.isdigit()), None)
            if SENSOR_ID:
                print(f"[INFO] Auto-detected sensor id: {SENSOR_ID}")
        raw_q.put((time.time(), data))
    except json.JSONDecodeError:
        pass

def parse_taxel_hex(hex_str: str):
    parts = hex_str.split(',')
    if len(parts) < TAXELS_EXPECTED * HEX_PER_TAXEL:
        return None
    try:
        return [int(h, 16) for h in parts[:TAXELS_EXPECTED*HEX_PER_TAXEL]]
    except ValueError:
        return None

def integrated_prompt_and_record(gui: PoseGUI):
    print("Recorder ready. Participant window is up. Starting in 3 s…")
    time.sleep(3)

    records = []

    for pose, dur in GESTURE_PLAN:
        gui.show_prepare(pose)
        for t in (5,4,3,2,1):
            gui.update_seconds(t)
            gui.update()
            time.sleep(1)

        gui.show_perform(pose)
        start = time.time()
        while time.time() - start < dur:
            sec_left = int(dur - (time.time() - start))
            gui.update_seconds(sec_left)
            gui.update()

            try:
                ts, pkt = raw_q.get(timeout=0.05)
            except queue.Empty:
                continue

            if SENSOR_ID is None or SENSOR_ID not in pkt:
                continue
            vals = parse_taxel_hex(pkt[SENSOR_ID].get("data", ""))
            if vals is None:
                continue
            records.append([ts, pose] + vals)

        print(f"Finished {pose}")

    gui.root.destroy()
    if not records:
        print("[WARN] No frames captured.")
        return None

    cols = ["t_sec","label"] + [f"x{r}{c}_{ax}"
            for r in range(4) for c in range(4) for ax in "xyz"]
    return pd.DataFrame(records, columns=cols)

def export(df: pd.DataFrame, prefix: str):
    if df is None or df.empty:
        print("[ERROR] Nothing to export.")
        return

    gesture_names = sorted({n for n,_ in GESTURE_PLAN})
    label_to_int  = {n:i for i,n in enumerate(gesture_names)}

    raw_csv  = f"{prefix}_raw.csv.gz"
    h5_file  = f"{prefix}_windows.h5"
    meta_pqt = f"{prefix}_meta.parquet"

    with gzip.open(raw_csv, "wt") as gz:
        df.to_csv(gz,index=False)

    step = int(STRIDE_MS / (1000 / SAMPLE_RATE))
    X,y,meta = [],[],[]
    for i in range(0, len(df)-T, step):
        win = df.iloc[i:i+T]
        lbl = win["label"].mode()[0]
        arr = win.iloc[:,2:].to_numpy(dtype=np.float32)\
               .reshape(T,48,1).reshape(T,16,3).transpose(2,1,0)\
               .reshape(3,4,4,T)
        X.append(arr); y.append(label_to_int[lbl])
        meta.append({"window_id":len(X)-1,"t_start":win["t_sec"].iloc[0],
                     "label_str":lbl})

    if not X:
        print("[ERROR] No windows generated.")
        return

    with h5py.File(h5_file,"w") as hf:
        hf.create_dataset("X",data=np.stack(X),compression="gzip",
                          chunks=(min(512,len(X)),)+X[0].shape)
        hf.create_dataset("y",data=np.array(y,dtype=np.uint8),
                          compression="gzip")
        hf.attrs["label_mapping"] = json.dumps(label_to_int)

    pd.DataFrame(meta).to_parquet(meta_pqt)
    print("✔ Exported:\n  •", raw_csv, "\n  •", h5_file, "\n  •", meta_pqt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record XELA data with on-screen pose prompts")
    parser.add_argument("-i", "--ip", required=True,
                        help="IP address of XELA sensor")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port number (default: 5000)")
    parser.add_argument("--sample-rate", type=int, default=100,
                        help="Sample rate in Hz (default: 100)")
    parser.add_argument("--window-ms", type=int, default=64,
                        help="Window size in milliseconds (default: 64)")
    parser.add_argument("--stride-ms", type=int, default=32,
                        help="Stride size in milliseconds (default: 32)")
    parser.add_argument("-p","--prefix", required=True,
                        help="File prefix (e.g. s01_trialA)")
    parser.add_argument("-g", "--gestures", default="gestures.yaml",
                        help="YAML file containing gesture plan (default: gestures.yaml)")
    args = parser.parse_args()

    # load gestures from yaml
    load_gesture_plan(args.gestures)

    IP = args.ip
    PORT = args.port
    SAMPLE_RATE = args.sample_rate
    WINDOW_MS = args.window_ms
    STRIDE_MS = args.stride_ms
    T = int(WINDOW_MS / (1000 / SAMPLE_RATE))

    ws = websocket.WebSocketApp(f"ws://{IP}:{PORT}", on_message=on_message)
    threading.Thread(target=ws.run_forever, daemon=True).start()

    gui = PoseGUI()
    df  = integrated_prompt_and_record(gui)
    export(df, args.prefix)
