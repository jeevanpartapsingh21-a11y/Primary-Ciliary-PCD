import os
import argparse
import numpy as np
import cv2
import pandas as pd

from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq


def video_to_array(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32))
        count += 1
        if max_frames is not None and count >= max_frames:
            break

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from: {video_path}")

    video_np_array = np.stack(frames, axis=0)  # (T, H, W)
    return video_np_array, fps

def split_video(video_matrix, n_split_axis):
    T, H, W = video_matrix.shape
    step_h = int(np.ceil(H / n_split_axis))
    step_w = int(np.ceil(W / n_split_axis))

    rois = []
    meta = []  # (roi_id, x, y, w, h)
    roi_id = 0

    for gy in range(n_split_axis):
        y0 = gy * step_h
        y1 = min((gy + 1) * step_h, H)
        for gx in range(n_split_axis):
            x0 = gx * step_w
            x1 = min((gx + 1) * step_w, W)
            roi = video_matrix[:, y0:y1, x0:x1]
            if roi.size == 0:
                continue
            rois.append(roi)
            meta.append((roi_id, x0, y0, x1 - x0, y1 - y0))
            roi_id += 1

    return rois, meta


def roi_mean_intensity(roi_cube):
    return roi_cube.mean(axis=(1, 2))


def center_signal(x):
    x = np.asarray(x, dtype=float)
    return x - np.mean(x)


def bandpass_filter(x, fs, low=2.0, high=30.0, order=2):
    nyq = 0.5 * fs
    low_cut = max(low / nyq, 1e-6)
    high_cut = min(high / nyq, 0.99)
    if low_cut >= high_cut:
        return x
    b, a = butter(order, [low_cut, high_cut], btype="band")
    return filtfilt(b, a, x)


def fft_peak(signal_1d, fs, fmin=2.0, fmax=30.0):
    s = np.asarray(signal_1d, dtype=float)
    if s.size < 8:
        return np.nan, np.nan, s

    s = center_signal(s)
    s_filt = bandpass_filter(s, fs, fmin, fmax, order=2)

    spec = rfft(s_filt)
    freqs = rfftfreq(len(s_filt), d=1.0 / fs)
    mags = np.abs(spec)

    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan, np.nan, s_filt

    idx = np.argmax(mags[mask])
    best_freq = float(freqs[mask][idx])
    best_mag = float(mags[mask][idx])

    return best_freq, best_mag, s_filt

def skeletonize_binary(binary_img):
    """
    Skeletonize using skimage if available, otherwise OpenCV thinning if available,
    otherwise return the binary image (still saves something).
    """
    b = (binary_img > 0).astype(np.uint8) * 255

    # option 1: skimage
    try:
        from skimage.morphology import skeletonize
        sk = skeletonize((b > 0))
        return (sk.astype(np.uint8) * 255)
    except Exception:
        pass

def save_skeleton_demo(frame_gray, out_path):
    """
    Takes ONE grayscale frame and makes a skeleton image.
    """
    g = frame_gray.astype(np.uint8)

    # Gaussian blur to reduce tiny noise
    g = cv2.GaussianBlur(g, (5, 5), 0)

    # threshold (invert because cilia often darker/bright depending on microscope)
    bin_img = cv2.adaptiveThreshold(
        g, 255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=5
    )

    # clean specks
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)

    skel = skeletonize_binary(bin_img)
    cv2.imwrite(out_path, skel)


def label_from_filename(filename):
    name = filename.lower()
    if "healthy" in name:
        return "Healthy"
    return "PCD"


#Adapted from https://github.com/SchubertLab/ciliated_cells_video_analysis/blob/main/notebooks/0-ciliary_beating_frequency_from_videos.ipynb
def process_videos_folder(video_dir, out_dir,
                        extensions=(".avi", ".mov", ".mp4", ".mkv"),
                        n_grid=10, max_frames=300, fmin=2.0, fmax=30.0):

    os.makedirs(out_dir, exist_ok=True)

    all_roi_rows = []
    all_video_rows = []

    files = sorted(os.listdir(video_dir))
    for fn in files:
        if not fn.lower().endswith(extensions):
            continue

        full_path = os.path.join(video_dir, fn)
        print("Processing:", full_path)

        try:
            frames, fps = video_to_array(full_path, max_frames=max_frames)
            T = frames.shape[0]

            mid = T // 2
            skel_path = os.path.join(out_dir, f"{os.path.splitext(fn)[0]}_skeleton.png")
            save_skeleton_demo(frames[mid], skel_path)

            # ROI features
            rois, meta = split_video(frames, n_grid)

            cbf_list = []
            for roi_cube, m in zip(rois, meta):
                roi_id, x, y, w, h = m
                sig = roi_mean_intensity(roi_cube)
                peak_f, peak_mag, filt = fft_peak(sig, fps, fmin=fmin, fmax=fmax)

                var_val = float(np.var(filt)) if filt.size else np.nan
                zcr_val = float(np.mean(np.diff(np.sign(filt)) != 0)) if filt.size > 1 else np.nan

                all_roi_rows.append({
                    "video": fn,
                    "label": label_from_filename(fn),
                    "fps": fps,
                    "roi_id": roi_id,
                    "x": x, "y": y, "w": w, "h": h,
                    "cbf_hz": peak_f,
                    "peak_mag": peak_mag,
                    "var": var_val,
                    "zcr": zcr_val,
                })

                if not np.isnan(peak_f):
                    cbf_list.append(peak_f)

            cbf_arr = np.array(cbf_list, dtype=float) if len(cbf_list) else np.array([])

            # per-video summary CBF
            all_video_rows.append({
                "video": fn,
                "label": label_from_filename(fn),
                "fps": fps,
                "n_rois": int(len(rois)),
                "n_rois_used": int(cbf_arr.size),
                "cbf_mean_hz": float(np.mean(cbf_arr)) if cbf_arr.size else np.nan,
                "cbf_median_hz": float(np.median(cbf_arr)) if cbf_arr.size else np.nan,
                "cbf_std_hz": float(np.std(cbf_arr)) if cbf_arr.size else np.nan,
                "cbf_min_hz": float(np.min(cbf_arr)) if cbf_arr.size else np.nan,
                "cbf_max_hz": float(np.max(cbf_arr)) if cbf_arr.size else np.nan,
            })

        except Exception as e:
            print("  Skipping", fn, "because:", e)

    # Save CSVs
    df_roi = pd.DataFrame(all_roi_rows)
    df_vid = pd.DataFrame(all_video_rows)

    roi_csv = os.path.join(out_dir, "cbf_per_roi.csv")
    vid_csv = os.path.join(out_dir, "cbf_per_video.csv")

    df_roi.to_csv(roi_csv, index=False)
    df_vid.to_csv(vid_csv, index=False)

    print("\nSaved CSV files:")
    print(" ", roi_csv)
    print(" ", vid_csv)
    print("Skeleton PNGs saved in:", out_dir)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    video_dir = os.path.join(script_dir, "data_videos")
    out_dir = os.path.join(script_dir, "output")

    print("Using videos from:", video_dir)
    print("Saving outputs to:", out_dir)

    process_videos_folder(
        video_dir=video_dir,
        out_dir=out_dir,
        extensions=(".avi", ".mov", ".mp4", ".mkv", ".MOV", ".AVI"),
        n_grid=10,
        max_frames=300,
        fmin=2.0,
        fmax=30.0,
    )