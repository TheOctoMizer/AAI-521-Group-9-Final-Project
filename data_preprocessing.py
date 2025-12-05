import cv2
import numpy as np
import random
from PIL import Image, ImageFilter, ImageEnhance
import os, h5py, cv2, numpy as np
from tqdm import tqdm

def degrade_image(img: np.ndarray):
    """Simulate old/damaged photo effects on an image."""
    h, w, _ = img.shape
    img = img.astype(np.float32) / 255.0

    # --- Blur / focus issues ---
    if random.random() < 0.5:
        k = random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # --- Add Gaussian noise ---
    if random.random() < 0.7:
        noise = np.random.normal(0, 0.04, img.shape)
        img = np.clip(img + noise, 0, 1)

    # --- JPEG compression artifacts ---
    if random.random() < 0.7:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(10, 60)]
        _, enc = cv2.imencode('.jpg', (img*255).astype(np.uint8), encode_param)
        img = cv2.imdecode(enc, 1) / 255.0

    # --- Fading / discoloration ---
    if random.random() < 0.5:
        factor = random.uniform(0.5, 1.0)
        img = img ** factor  # gamma correction (fading)

    # --- Scratches / spots ---
    if random.random() < 0.3:
        mask = np.ones((h, w, 1), dtype=np.float32)
        for _ in range(random.randint(1, 5)):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = x1 + np.random.randint(10, 50), y1 + np.random.randint(1, 3)
            cv2.line(mask, (x1, y1), (x2, y2), (0,), thickness=1)
        img = img * mask

    # --- Color shift ---
    if random.random() < 0.3:
        shift = np.random.uniform(-0.1, 0.1, (1, 3))
        img = np.clip(img + shift, 0, 1)

    # --- Sepia tone ---
    if random.random() < 0.3:
        img = np.clip(img * np.array([0.393, 0.769, 0.189]), 0, 1)

    # --- Random Occlusion ---
    if random.random() < 0.3:
        mask = np.ones((h, w, 1), dtype=np.float32)
        for _ in range(random.randint(1, 5)):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = x1 + np.random.randint(10, 50), y1 + np.random.randint(1, 3)
            cv2.line(mask, (x1, y1), (x2, y2), (0,), thickness=1)
        img = img * mask
    
    # --- Random Deformation ---
    if random.random() < 0.3:
        mask = np.ones((h, w, 1), dtype=np.float32)
        for _ in range(random.randint(1, 5)):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = x1 + np.random.randint(10, 50), y1 + np.random.randint(1, 3)
            cv2.line(mask, (x1, y1), (x2, y2), (0,), thickness=1)
        img = img * mask

    # --- Random Cracks ---
    if random.random() < 0.3:
        mask = np.ones((h, w, 1), dtype=np.float32)
        for _ in range(random.randint(1, 5)):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = x1 + np.random.randint(10, 50), y1 + np.random.randint(1, 3)
            cv2.line(mask, (x1, y1), (x2, y2), (0,), thickness=1)
        img = img * mask

    # --- Convert back to uint8 ---
    return (img * 255).astype(np.uint8)

src_dir = "raw_images"
h5_path = "dataset.h5"

img_files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

with h5py.File(h5_path, "w") as f:
    grp_input = f.create_group("input")
    grp_target = f.create_group("target")

    for i, fname in enumerate(tqdm(img_files)):
        path = os.path.join(src_dir, fname)
        clean = cv2.imread(path)
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        degraded = degrade_image(clean)

        # store as uint8 to save space
        grp_input.create_dataset(str(i), data=degraded, compression="gzip")
        grp_target.create_dataset(str(i), data=clean, compression="gzip")

