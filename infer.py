"""
Example
-------
python xela_live_infer_gui.py \
    --model-ckpt  personalized_xela.pth \
    --stats       stats.npz \
    --h5-ref      training_session1.h5 \
    --ip 192.168.11.5 --port 5000 \
    --switch-threshold 0.85
"""
import argparse, json, queue, threading, sys
from collections import deque
from pathlib import Path

import numpy as np, h5py, torch, torch.nn.functional as F
import websocket
import tkinter as tk

DISPLAY_MAP = {
    "ulnar_dev" : "ulnar deviation",
    "uldar_dev" : "ulnar deviation",   # typo-tolerant alias
    "radial_dev": "radial deviation",
}

HEX_PER_TAXEL  = 3
TAXELS_EXPECTED = 16

def parse_taxel_hex(hex_str: str):
    parts = hex_str.split(',')
    if len(parts) < TAXELS_EXPECTED * HEX_PER_TAXEL:
        return None
    try:
        return [int(h, 16) for h in parts[:TAXELS_EXPECTED*HEX_PER_TAXEL]]
    except ValueError:
        return None

class XelaStream:
    """Pull raw frames from sensor websocket; yields numpy (48,) xyz*16."""
    def __init__(self, ip: str, port: int):
        self.url   = f"ws://{ip}:{port}"
        self.raw_q = queue.Queue()
        self.sensor_id = None
        self.ws = websocket.WebSocketApp(
            self.url,
            on_message=self.on_message,
            on_close=lambda *_: print("WebSocket closed"))
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def on_message(self, _wsapp, message):
        try:
            data = json.loads(message)
            if data.get("type") == "welcome":
                return
            if self.sensor_id is None:
                for k in data.keys():
                    if k.isdigit():
                        self.sensor_id = k
                        print(f"[INFO] sensor_id detected: {self.sensor_id}")
                        break
            if self.sensor_id is None or self.sensor_id not in data:
                return
            vals = parse_taxel_hex(data[self.sensor_id].get("data", ""))
            if vals is not None:
                self.raw_q.put(np.array(vals, dtype=np.float32))
        except json.JSONDecodeError:
            pass

    def get_frame(self, timeout=1.0):
        return self.raw_q.get(timeout=timeout)

class FullscreenDisplay:
    BLUE = "#1E66FF"
    def __init__(self, title="XELA Gesture Inference"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg="white")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self.text_var = tk.StringVar(value="WAITING")
        self.label = tk.Label(self.root, textvariable=self.text_var,
                              fg=self.BLUE, bg="white",
                              font=("Helvetica", 200, "bold"))
        self.label.pack(expand=True, fill="both")
        self.root.update_idletasks()
        self._resize_font()
        self.root.bind("<Configure>", lambda e: self._resize_font())

    def _resize_font(self):
        w, h = self.root.winfo_width(), self.root.winfo_height()
        size = max(20, int(min(w, h) * 0.10))
        self.label.config(font=("Helvetica", size, "bold"))

    def set_text(self, txt: str):
        self.text_var.set(txt.upper())

    def start(self):
        self.root.mainloop()

def inference_loop(args, update_q: queue.Queue):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with h5py.File(args.h5_ref, "r") as hf:
        label_map = json.loads(hf.attrs["label_mapping"])
        T = hf["X"].shape[-1]
    id2label = {v: k for k, v in label_map.items()}
    print(f"[INFO] Loaded classes: {list(id2label.values())} | T={T}")

    stats = np.load(args.stats)
    mean, std = stats["mean"], stats["std"]

    from model.triforcemodel import SpatialMixerXS
    in_ch = 1 if args.normal_only else 3
    model = SpatialMixerXS(len(id2label), in_ch=in_ch, dropout=0.0).to(device)
    model.load_state_dict(torch.load(args.model_ckpt, map_location="cpu"), strict=False)
    model.eval()
    print(f"[INFO] Model loaded from {args.model_ckpt}")

    buf = deque(maxlen=T)
    stream = XelaStream(args.ip, args.port)
    current_label = None

    while True:
        raw = stream.get_frame()
        frame = raw.reshape(16, 3).T.reshape(3, 4, 4)
        if args.normal_only:
            frame = frame[2:3]
        buf.append(frame)
        if len(buf) < T:
            continue

        x = np.stack(buf, axis=-1).mean(-1).astype(np.float32)   # (C,4,4)
        x = (x - mean) / (std + 1e-8)
        tensor = torch.from_numpy(x).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1).cpu().squeeze(0).numpy()

        pred_id = int(probs.argmax())
        pred_lbl = id2label[pred_id]
        conf = float(probs[pred_id])

        if (pred_lbl != current_label) and (conf >= args.switch_threshold):
            current_label = pred_lbl
            pretty = DISPLAY_MAP.get(current_label, current_label)

            if update_q.full():
                try: update_q.get_nowait()
                except queue.Empty: pass
            update_q.put_nowait(pretty)

def main(args):
    update_q: queue.Queue[str] = queue.Queue(maxsize=1)
    threading.Thread(target=inference_loop, args=(args, update_q), daemon=True).start()

    gui = FullscreenDisplay()

    def gui_poll():
        try:
            gui.set_text(update_q.get_nowait())
        except queue.Empty:
            pass
        gui.root.after(50, gui_poll)

    gui_poll()
    gui.start()

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Live XELA gesture inference with SpatialMixer-XS + full-screen GUI")
    p.add_argument("--model-ckpt", required=True)
    p.add_argument("--stats",      required=True)
    p.add_argument("--h5-ref",     required=True)
    p.add_argument("--ip",   default="10.20.82.60")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--normal-only", action="store_true",
                   help="Use Z channel only (must match training).")
    p.add_argument("--switch-threshold", type=float, default=0.70,
                   help="Probability required to switch to a new class.")
    args = p.parse_args()
    main(args)
