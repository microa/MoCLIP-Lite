#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, argparse, numpy as np
from pathlib import Path
from PIL import Image
from multiprocessing import Pool, cpu_count

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback

NUMERIC_RE = re.compile(r".*?(\d+)\.npy$")

def list_mv_files(video_dir):
    files = []
    for name in os.listdir(video_dir):
        if name.endswith(".npy"):
            m = NUMERIC_RE.match(name)
            if m:
                files.append((int(m.group(1)), os.path.join(video_dir, name)))
    files.sort(key=lambda x: x[0])
    return files

def hsv_to_rgb_numpy(h, s, v):
    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = np.zeros_like(h); g = np.zeros_like(h); b = np.zeros_like(h)
    i_mod = i % 6
    mask = (i_mod == 0); r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]
    mask = (i_mod == 1); r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]
    mask = (i_mod == 2); r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]
    mask = (i_mod == 3); r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]
    mask = (i_mod == 4); r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]
    mask = (i_mod == 5); r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]
    return np.stack([r, g, b], axis=-1)

def infer_grid_and_block_from_first_nonempty(mv_paths):
    for _, p in mv_paths:
        arr = np.load(p, allow_pickle=False)
        if arr.size == 0: 
            continue
        bw = int(np.min(arr[:, 1])); bh = int(np.min(arr[:, 2]))
        max_x = np.max(arr[:, 3] + arr[:, 1]); max_y = np.max(arr[:, 4] + arr[:, 2])
        grid_w = int(np.ceil(max_x / bw)); grid_h = int(np.ceil(max_y / bh))
        scale_default = int(np.median(arr[:, 9])) if arr.shape[1] > 9 else 1
        return grid_h, grid_w, bw, bh, max(scale_default, 1)
    return 1, 1, 16, 16, 1  # all-empty fallback

def mv_array_to_dense_field(arr, grid_h, grid_w, base_w, base_h, stabilize=False):
    dx = np.zeros((grid_h, grid_w), np.float32)
    dy = np.zeros((grid_h, grid_w), np.float32)
    cnt = np.zeros((grid_h, grid_w), np.int32)
    if arr.size == 0: 
        return dx, dy
    sx = arr[:, 3].astype(float); sy = arr[:, 4].astype(float)
    mvx = arr[:, 7].astype(float); mvy = arr[:, 8].astype(float)
    scale = arr[:, 9].astype(float) if arr.shape[1] > 9 else np.ones_like(mvx)
    scale[scale == 0] = 1.0
    mvx = mvx / scale; mvy = mvy / scale
    j = np.clip((sx / base_w).astype(int), 0, grid_w - 1)
    i = np.clip((sy / base_h).astype(int), 0, grid_h - 1)
    np.add.at(dx, (i, j), mvx); np.add.at(dy, (i, j), mvy); np.add.at(cnt, (i, j), 1)
    mask = cnt > 0
    dx[mask] /= cnt[mask]; dy[mask] /= cnt[mask]
    if stabilize and np.any(mask):
        dx -= np.median(dx[mask]); dy -= np.median(dy[mask])
    return dx, dy

def field_to_rgb(dx, dy, encode="hsv", robust_percentile=95):
    mag = np.sqrt(dx*dx + dy*dy)
    if encode == "hsv":
        angle = np.arctan2(dy, dx)
        h = (angle + np.pi) / (2*np.pi)
        s = np.ones_like(h, np.float32)
        vmax = max(np.percentile(mag, robust_percentile), 1e-6) if mag.size else 1.0
        v = np.clip(mag / vmax, 0.0, 1.0)
        rgb = hsv_to_rgb_numpy(h, s, v)
        return (rgb * 255.0 + 0.5).astype(np.uint8)
    elif encode == "dxdy_mag":
        lim = max(np.percentile(np.abs(np.concatenate([dx.ravel(), dy.ravel()])), robust_percentile), 1e-6)
        ch0 = np.clip((dx/lim + 1)/2, 0, 1); ch1 = np.clip((dy/lim + 1)/2, 0, 1)
        mmax = max(np.percentile(mag, robust_percentile), 1e-6)
        ch2 = np.clip(mag / mmax, 0, 1)
        img = np.stack([ch0, ch1, ch2], -1)
        return (img * 255.0 + 0.5).astype(np.uint8)
    else:
        raise ValueError("encode must be hsv or dxdy_mag")

def resize_uint8(img, size):
    if size is None: 
        return img
    return np.array(Image.fromarray(img).resize((size, size), resample=Image.NEAREST))

def process_video_motion_dir(mv_dir, out_dir, mode="per_frame", size=224,
                             encode="hsv", stabilize=False, save_stack_npy=False):
    os.makedirs(out_dir, exist_ok=True)
    mv_paths = list_mv_files(mv_dir)
    if not mv_paths:
        return {"video": str(Path(mv_dir).parent), "frames": 0, "status": "no_npy"}
    grid_h, grid_w, bw, bh, _ = infer_grid_and_block_from_first_nonempty(mv_paths)
    stack = []
    frames_written = 0

    for fid, fpath in mv_paths:
        arr = np.load(fpath, allow_pickle=False)
        dx, dy = mv_array_to_dense_field(arr, grid_h, grid_w, bw, bh, stabilize=stabilize)
        img = field_to_rgb(dx, dy, encode=encode, robust_percentile=95)
        img = resize_uint8(img, size)
        if mode == "per_frame":
            Image.fromarray(img).save(os.path.join(out_dir, f"mv_rgb_{fid:05d}.png"))
            frames_written += 1
        else:
            stack.append(img)

    if mode.startswith("collapse"):
        if stack:
            arr = np.stack(stack, 0)
            collapsed = (arr.mean(0) if mode == "collapse_mean" else arr.max(0)).astype(np.uint8)
        else:
            collapsed = np.zeros((size if size else grid_h, size if size else grid_w, 3), np.uint8)
        Image.fromarray(collapsed).save(os.path.join(out_dir, f"mv_rgb_{mode}.png"))

    if save_stack_npy:
        # 将整段按时间堆叠保存，便于后续直接加载为 (T,H,W,3)
        if mode == "per_frame":
            # 重新收集以避免双倍内存：按需从磁盘读回；或简单再次遍历（这里直接再次遍历）
            buf = []
            for fid, fpath in mv_paths:
                png_path = os.path.join(out_dir, f"mv_rgb_{fid:05d}.png")
                if os.path.exists(png_path):
                    buf.append(np.array(Image.open(png_path)))
            if buf:
                np.save(os.path.join(out_dir, "mv_rgb_stack.npy"), np.stack(buf, 0))
        else:
            if stack:
                np.save(os.path.join(out_dir, "mv_rgb_stack.npy"), np.stack(stack, 0))

    return {"video": str(Path(mv_dir).parent), "frames": frames_written, "status": "ok"}

def find_all_motion_dirs(dataset_root):
    motion_dirs = []
    for root, dirs, files in os.walk(dataset_root):
        if os.path.basename(root) == "motion_vectors":
            # 简单校验：里面有 .npy
            if any(fn.endswith(".npy") for fn in files):
                motion_dirs.append(root)
    return motion_dirs

def _worker(args):
    mv_dir, mode, size, encode, stabilize, save_stack_npy = args
    video_dir = str(Path(mv_dir).parent)
    out_dir = os.path.join(video_dir, "mv_rgb")
    try:
        return process_video_motion_dir(
            mv_dir, out_dir, mode=mode, size=size,
            encode=encode, stabilize=stabilize, save_stack_npy=save_stack_npy
        )
    except Exception as e:
        return {"video": video_dir, "frames": 0, "status": f"error: {e}"}

def main():
    ap = argparse.ArgumentParser()
    # 单视频 or 整库二选一
    ap.add_argument("--video_dir", type=str, help="Path to a single motion_vectors directory")
    ap.add_argument("--out_dir", type=str, help="Output dir for single-video mode (default=<video>/mv_rgb)")
    ap.add_argument("--dataset_root", type=str, help="Root containing */Class/Video/motion_vectors/*.npy")
    ap.add_argument("--mode", type=str, default="per_frame", choices=["per_frame","collapse_mean","collapse_max"])
    ap.add_argument("--size", type=int, default=224, help="Output resolution (square). 0=keep grid size")
    ap.add_argument("--encode", type=str, default="hsv", choices=["hsv","dxdy_mag"])
    ap.add_argument("--stabilize", action="store_true", help="Subtract per-frame median motion (camera shake)")
    ap.add_argument("--save_stack_npy", action="store_true", help="Also save (T,H,W,3) uint8 stack as mv_rgb_stack.npy")
    ap.add_argument("--workers", type=int, default=max(cpu_count()//2, 1), help="Multiprocessing workers for dataset mode")
    args = ap.parse_args()

    size = None if args.size == 0 else args.size

    if args.dataset_root:
        motion_dirs = find_all_motion_dirs(args.dataset_root)
        if not motion_dirs:
            print("No motion_vectors directories found under dataset_root.")
            return
        work = [(mv, args.mode, size, args.encode, args.stabilize, args.save_stack_npy) for mv in motion_dirs]
        with Pool(processes=max(1, args.workers)) as pool:
            for res in tqdm(pool.imap_unordered(_worker, work), total=len(work)):
                print(f"[{res['status']}] {res['video']} ({res['frames']} frames)")
    elif args.video_dir:
        out_dir = args.out_dir or os.path.join(str(Path(args.video_dir).parent), "mv_rgb")
        res = process_video_motion_dir(
            args.video_dir, out_dir, mode=args.mode, size=size,
            encode=args.encode, stabilize=args.stabilize, save_stack_npy=args.save_stack_npy
        )
        print(f"[{res['status']}] {res['video']} ({res['frames']} frames) -> {out_dir}")
    else:
        raise SystemExit("Please specify --dataset_root or --video_dir.")

if __name__ == "__main__":
    main()
