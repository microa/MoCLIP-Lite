#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, argparse, numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUMERIC_RE = re.compile(r".*?(\d+)\.npy$")

def list_mv_files(video_dir):
    items = []
    for n in os.listdir(video_dir):
        if n.endswith(".npy"):
            m = NUMERIC_RE.match(n)
            if m: items.append((int(m.group(1)), os.path.join(video_dir, n)))
    items.sort(key=lambda x: x[0])
    return items

def infer_grid(mv_paths):
    for _, p in mv_paths:
        arr = np.load(p, allow_pickle=False)
        if arr.size == 0: continue
        bw = int(np.min(arr[:,1])); bh = int(np.min(arr[:,2]))
        max_x = np.max(arr[:,3] + arr[:,1]); max_y = np.max(arr[:,4] + arr[:,2])
        gw = int(np.ceil(max_x / bw)); gh = int(np.ceil(max_y / bh))
        return gh, gw, bw, bh
    return 1,1,16,16

def mv_to_field(arr, gh, gw, bw, bh, stabilize=False):
    dx = np.zeros((gh,gw), np.float32)
    dy = np.zeros((gh,gw), np.float32)
    cnt = np.zeros((gh,gw), np.int32)
    if arr.size == 0: return dx,dy
    sx = arr[:,3].astype(float); sy = arr[:,4].astype(float)
    mvx = arr[:,7].astype(float); mvy = arr[:,8].astype(float)
    scale = arr[:,9].astype(float) if arr.shape[1]>9 else np.ones_like(mvx)
    scale[scale==0]=1.0
    mvx /= scale; mvy /= scale
    j = np.clip((sx / bw).astype(int), 0, gw-1)
    i = np.clip((sy / bh).astype(int), 0, gh-1)
    np.add.at(dx, (i,j), mvx); np.add.at(dy, (i,j), mvy); np.add.at(cnt, (i,j), 1)
    mask = cnt>0
    dx[mask] /= cnt[mask]; dy[mask] /= cnt[mask]
    if stabilize and np.any(mask):
        dx -= np.median(dx[mask]); dy -= np.median(dy[mask])
    return dx,dy

def angle_to_color(dx, dy):
    ang = np.arctan2(dy, dx)  # [-pi,pi]
    hue = (ang + np.pi) / (2*np.pi)  # [0,1]
    cmap = plt.get_cmap("hsv")
    col = cmap(hue)  # RGBA
    return col

def draw_quiver(dx, dy, step=1, scale=1.0, save_path="out.png"):
    H,W = dx.shape
    yy, xx = np.mgrid[0:H:1, 0:W:1]
    # 下采样
    xx = xx[::step, ::step]; yy = yy[::step, ::step]
    u  =  dx[::step, ::step]; v  = -dy[::step, ::step]  # 画图坐标y轴向下，取反更直观
    colors = angle_to_color(u, -v)  # 用方向上色

    fig = plt.figure(figsize=(W/6, H/6), dpi=100)  # 自适比例
    ax = fig.add_axes([0,0,1,1]); ax.set_axis_off(); ax.set_xlim(-0.5, W-0.5); ax.set_ylim(H-0.5, -0.5)
    ax.set_facecolor("black")
    # scale与width可按需调节
    q = ax.quiver(xx, yy, u, v, angles='xy', scale_units='xy', scale=1.0/scale, width=0.006, color=colors)
    fig.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def process_video(mv_dir, out_dir, step=1, scale=1.0, stabilize=False):
    os.makedirs(out_dir, exist_ok=True)
    mv_paths = list_mv_files(mv_dir)
    if not mv_paths: return (str(Path(mv_dir).parent), 0, "no_npy")
    gh, gw, bw, bh = infer_grid(mv_paths)
    n = 0
    for fid, fpath in mv_paths:
        arr = np.load(fpath, allow_pickle=False)
        dx, dy = mv_to_field(arr, gh, gw, bw, bh, stabilize=stabilize)
        save_path = os.path.join(out_dir, f"mv_quiver_{fid:05d}.png")
        draw_quiver(dx, dy, step=step, scale=scale, save_path=save_path)
        n += 1
    return (str(Path(mv_dir).parent), n, "ok")

def find_motion_dirs(root):
    hits=[]
    for r, d, f in os.walk(root):
        if os.path.basename(r)=="motion_vectors" and any(x.endswith(".npy") for x in f):
            hits.append(r)
    return hits

def _worker(args):
    mv_dir, out_subdir, step, scale, stabilize = args
    video_dir = str(Path(mv_dir).parent)
    out_dir = os.path.join(video_dir, out_subdir)
    try:
        return process_video(mv_dir, out_dir, step=step, scale=scale, stabilize=stabilize)
    except Exception as e:
        return (video_dir, 0, f"error: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", type=str)
    ap.add_argument("--dataset_root", type=str)
    ap.add_argument("--out_subdir", type=str, default="mv_quiver")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--stabilize", action="store_true")
    ap.add_argument("--workers", type=int, default=max(cpu_count()//2,1))
    args = ap.parse_args()

    if args.dataset_root:
        dirs = find_motion_dirs(args.dataset_root)
        work = [(d, args.out_subdir, args.step, args.scale, args.stabilize) for d in dirs]
        with Pool(processes=args.workers) as pool:
            for video, n, st in pool.imap_unordered(_worker, work):
                print(f"[{st}] {video} ({n} frames)")
    elif args.video_dir:
        video = str(Path(args.video_dir).parent)
        out_dir = os.path.join(video, args.out_subdir)
        video, n, st = process_video(args.video_dir, out_dir, step=args.step, scale=args.scale, stabilize=args.stabilize)
        print(f"[{st}] {video} ({n} frames) -> {out_dir}")
    else:
        raise SystemExit("Specify --dataset_root or --video_dir")

if __name__ == "__main__":
    main()
