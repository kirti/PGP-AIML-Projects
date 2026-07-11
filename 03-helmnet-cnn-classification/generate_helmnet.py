"""
Synthetic HelmNet image dataset generator (v2 - more photo-like texture).
Produces images_proj.npy (631, 200, 200, 3) uint8 and Labels_proj.csv,
matching the real dataset's shape/class-balance, WITHOUT using any real photos.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import pandas as pd

rng = np.random.default_rng(42)
N_HELMET = 311
N_NO_HELMET = 320
IMG_SIZE = 200

def make_textured_background(rng):
    # per-pixel noise (not just a flat color + tiny noise) so the model sees realistic texture
    base = rng.integers(70, 170)
    noise = rng.normal(0, 35, size=(IMG_SIZE, IMG_SIZE, 3))
    tint = rng.integers(-20, 20, size=3)
    img = base + noise + tint
    img = np.clip(img, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=1.5))
    return pil_img

def make_helmet_image(rng):
    img = make_textured_background(rng)
    draw = ImageDraw.Draw(img)
    cx = rng.integers(70, 130)
    cy = rng.integers(40, 80)
    r = rng.integers(25, 40)
    helmet_color = tuple(int(v) for v in rng.integers(150, 255, size=3))
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=helmet_color, outline=(0,0,0))
    draw.rectangle([cx-r-5, cy+r//2, cx+r+5, cy+r//2+8], fill=helmet_color)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    arr = np.array(img).astype(np.int16)
    arr += rng.integers(-15, 15, size=arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)

def make_no_helmet_image(rng):
    img = make_textured_background(rng)
    draw = ImageDraw.Draw(img)
    cx = rng.integers(70, 130)
    cy = rng.integers(50, 100)
    r = rng.integers(20, 30)
    skin_color = tuple(int(v) for v in rng.integers(120, 200, size=3))
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=skin_color, outline=(80,60,40))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    arr = np.array(img).astype(np.int16)
    arr += rng.integers(-15, 15, size=arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)

images = []
labels = []
for _ in range(N_HELMET):
    images.append(make_helmet_image(rng)); labels.append(1)
for _ in range(N_NO_HELMET):
    images.append(make_no_helmet_image(rng)); labels.append(0)

images = np.array(images, dtype=np.uint8)
labels_df = pd.DataFrame({'label': labels})
perm = rng.permutation(len(images))
images = images[perm]
labels_df = labels_df.iloc[perm].reset_index(drop=True)

np.save('/mnt/user-data/outputs/images_proj_DUMMY.npy', images)
labels_df.to_csv('/mnt/user-data/outputs/Labels_proj_DUMMY.csv', index=False)
print("images shape:", images.shape)
print(labels_df['label'].value_counts())
print("pixel std (texture check):", images.astype(float).std())
