"""NIH ChestX-ray14 data pipeline: manifest, patient-wise split, image cache, datasets."""
import glob
import os
import random
from multiprocessing import Pool

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import LABELS, wpath

BBOX_NAME_MAP = {"Infiltrate": "Infiltration"}


# ----------------------------------------------------------------------------- locating files
def _find_one(root, name):
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    if not hits:
        raise FileNotFoundError(f"'{name}' not found under {root}. Set data.nih_root correctly.")
    return sorted(hits, key=len)[0]


def index_images(root):
    """Map image file name -> full path (NIH images live in images_001/images ... images_012/images)."""
    paths = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".png"):
                paths[f] = os.path.join(dirpath, f)
    if not paths:
        raise FileNotFoundError(f"No PNG images found under {root}")
    return paths


# ----------------------------------------------------------------------------- manifest
def build_manifest(cfg, image_paths=None):
    import pandas as pd
    d = cfg["data"]
    root = d["nih_root"]
    entry = pd.read_csv(_find_one(root, "Data_Entry_2017*.csv"))
    entry.columns = [c.strip() for c in entry.columns]
    with open(_find_one(root, "train_val_list*.txt")) as f:
        train_val = {l.strip() for l in f if l.strip()}
    with open(_find_one(root, "test_list*.txt")) as f:
        test = {l.strip() for l in f if l.strip()}
    paths = index_images(root) if image_paths is None else image_paths

    df = pd.DataFrame({
        "image": entry["Image Index"],
        "patient": entry["Patient ID"].astype(int),
        "age": pd.to_numeric(entry["Patient Age"], errors="coerce"),
        "sex": entry["Patient Gender"].astype(str).str.upper().str[0],
        "view": entry["View Position"].astype(str).str.upper(),
        "follow_up": entry["Follow-up #"].astype(int),
    })
    df.loc[(df.age < 0) | (df.age > 100), "age"] = np.nan        # a few NIH ages are corrupt (e.g. 414)
    wcol = [c for c in entry.columns if "Width" in c]
    hcol = [c for c in entry.columns if c.startswith("Height")]
    df["orig_w"] = entry[wcol[0]].astype(int) if wcol else 1024
    df["orig_h"] = entry[hcol[0]].astype(int) if hcol else 1024
    findings = entry["Finding Labels"].astype(str).str.split("|")
    for lab in LABELS:
        df[lab] = findings.apply(lambda fs, l=lab: int(l in fs)).astype(np.int8)
    df["path"] = df.image.map(paths)
    missing = df.path.isna().sum()
    if missing:
        print(f"[manifest] warning: {missing} images listed in Data_Entry are missing on disk -> dropped")
        df = df[df.path.notna()].copy()

    df["split"] = "unused"
    df.loc[df.image.isin(test), "split"] = "test"
    tv = df.image.isin(train_val)
    # patient-wise validation split (no patient appears in two splits -> no leakage)
    if d.get("val_list"):
        with open(d["val_list"]) as file:
            val_images = {line.strip() for line in file if line.strip()}
        if not val_images <= train_val or not val_images <= set(df.image):
            raise ValueError("Validation list must contain available images from the official training pool")
        val_pats = set(df.loc[df.image.isin(val_images), "patient"])
    else:
        rng = np.random.RandomState(cfg["seed"])
        pats = np.array(sorted(df.loc[tv, "patient"].unique()))
        rng.shuffle(pats)
        val_pats = set(pats[: int(round(len(pats) * d["val_frac"]))])
    df.loc[tv, "split"] = np.where(df.loc[tv, "patient"].isin(val_pats), "val", "train")
    df = df[df.split != "unused"].reset_index(drop=True)
    assert not (set(df[df.split == "train"].patient) & set(df[df.split == "test"].patient)), "patient leakage"
    assert not (set(df[df.split == "train"].patient) & set(df[df.split == "val"].patient)), "patient leakage"
    assert not (set(df[df.split == "val"].patient) & set(df[df.split == "test"].patient)), "patient leakage"
    df["idx"] = np.arange(len(df))
    df.to_csv(wpath(cfg, "manifest.csv"), index=False)

    # bounding boxes (984 radiologist boxes, 8 classes) for localisation evaluation
    try:
        bb = pd.read_csv(_find_one(root, "BBox_List_2017*.csv"))
        bb = bb.iloc[:, :6]
        bb.columns = ["image", "label", "x", "y", "w", "h"]
        bb["label"] = bb.label.replace(BBOX_NAME_MAP)
        bb = bb.merge(df[["image", "idx", "split"]], on="image", how="inner")
        bb.to_csv(wpath(cfg, "bbox.csv"), index=False)
        print(f"[manifest] {len(bb)} bounding boxes ({bb.split.value_counts().to_dict()})")
    except FileNotFoundError as e:
        print("[manifest] no bounding-box file:", e)

    print("[manifest] split sizes:", df.split.value_counts().to_dict())
    print("[manifest] patients:", df.groupby("split").patient.nunique().to_dict())
    prev = df[df.split == "train"][LABELS].mean().round(4)
    print("[manifest] train prevalence:\n" + prev.to_string())
    return df


def load_manifest(cfg):
    import pandas as pd
    p = wpath(cfg, "manifest.csv")
    if not os.path.exists(p):
        raise FileNotFoundError("manifest.csv missing - run:  python run.py manifest")
    return pd.read_csv(p)


# ----------------------------------------------------------------------------- image cache
def _load_resize(args):
    path, size = args
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return np.zeros((size, size), np.uint8), False
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA), True


def cache_path(cfg):
    d = cfg["data"]
    return os.path.join(d["cache_dir"], f"nih_{d['cache_size']}.u8")


def build_cache(cfg, workers=None):
    """Decode + resize every image once into a uint8 memmap. PNG decoding of 1024x1024 images is the
    main bottleneck of naive training (see the ~2-4 s/iteration in the CPU logs); afterwards an epoch
    is GPU-bound."""
    df = load_manifest(cfg)
    size = cfg["data"]["cache_size"]
    os.makedirs(cfg["data"]["cache_dir"], exist_ok=True)
    out = cache_path(cfg)
    mm = np.lib.format.open_memmap(out + ".npy", mode="w+", dtype=np.uint8, shape=(len(df), size, size))
    workers = workers or max(1, (os.cpu_count() or 2))
    bad = 0
    from tqdm.auto import tqdm
    with Pool(workers) as pool:
        it = pool.imap(_load_resize, [(p, size) for p in df.path], chunksize=64)
        for i, (img, ok) in enumerate(tqdm(it, total=len(df), desc=f"cache {size}px")):
            mm[i] = img
            bad += (not ok)
    mm.flush()
    del mm
    print(f"[cache] wrote {out}.npy  ({len(df)} images, {bad} unreadable)")


def open_cache(cfg):
    p = cache_path(cfg) + ".npy"
    if cfg["data"].get("use_cache", True) and os.path.exists(p):
        return np.load(p, mmap_mode="r")
    return None


# ----------------------------------------------------------------------------- transforms
_CLAHE = None


def apply_clahe(g):
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _CLAHE.apply(g)


def train_augment(g, size, rot_deg=7.0, scale_min=0.8):
    """Random-resized crop + small rotation + brightness/contrast jitter.
    No horizontal flip: it would swap the patient's left and right, and laterality is part of the report."""
    h, w = g.shape
    s = random.uniform(scale_min, 1.0)
    r = random.uniform(0.9, 1.1)
    ch, cw = int(min(h, h * np.sqrt(s / r))), int(min(w, w * np.sqrt(s * r)))
    y0, x0 = random.randint(0, h - ch), random.randint(0, w - cw)
    g = g[y0:y0 + ch, x0:x0 + cw]
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    M = cv2.getRotationMatrix2D((size / 2, size / 2), random.uniform(-rot_deg, rot_deg), 1.0)
    g = cv2.warpAffine(g, M, (size, size), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a, b = random.uniform(0.9, 1.1), random.uniform(-12, 12)
    return np.clip(g.astype(np.float32) * a + b, 0, 255).astype(np.uint8)


def eval_transform(g, size):
    """Full-field resize (no centre crop) so CAM coordinates map linearly back to the original image."""
    if g.shape != (size, size):
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return g


def to_gray_uint8(img):
    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = (255 * (img - img.min()) / max(1e-6, float(img.max() - img.min()))).astype(np.uint8)
    return img


# ----------------------------------------------------------------------------- dataset
class CXRDataset(Dataset):
    """Returns (uint8 tensor 1xHxW, float labels, row index). Normalisation happens on the GPU inside the model."""

    def __init__(self, df, cfg, train=False, cache=None, img_size=None):
        self.df = df.reset_index(drop=True)
        self.idx = self.df["idx"].values
        self.paths = self.df["path"].values
        self.y = self.df[LABELS].values.astype(np.float32)
        self.cache = cache
        self.cache_size = cfg["data"]["cache_size"]
        self.size = img_size or cfg["train"]["img_size"]
        self.train = train
        self.clahe = cfg["data"].get("clahe", False)
        t = cfg["train"]
        self.rot, self.smin = t["rot_deg"], t["crop_scale_min"]

    def __len__(self):
        return len(self.df)

    def _image(self, i):
        if self.cache is not None:
            return np.asarray(self.cache[self.idx[i]])
        g = cv2.imread(self.paths[i], cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise FileNotFoundError(self.paths[i])
        return cv2.resize(g, (self.cache_size, self.cache_size), interpolation=cv2.INTER_AREA)

    def __getitem__(self, i):
        g = self._image(i)
        if self.clahe:
            g = apply_clahe(g)
        g = train_augment(g, self.size, self.rot, self.smin) if self.train else eval_transform(g, self.size)
        return torch.from_numpy(np.ascontiguousarray(g))[None], torch.from_numpy(self.y[i]), int(self.idx[i])


def worker_init(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)
