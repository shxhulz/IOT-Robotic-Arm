import os
import urllib.request
import torch
import cv2
import numpy as np

ckpt = "sam2_hiera_large.pt"
if not os.path.exists(ckpt):
    print("Downloading checkpoint...")
    urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt", ckpt)

print("Checkpoint ready.")

os.makedirs("test_imgs", exist_ok=True)
for i in range(4):
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 320), 100, (255, 255, 255), -1)
    cv2.imwrite(f"test_imgs/test_{i}.jpg", img)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_sam2("sam2_hiera_l.yaml", ckpt, device=device)
mask_gen = SAM2AutomaticMaskGenerator(model)

images = []
for i in range(4):
    img = cv2.imread(f"test_imgs/test_{i}.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)

print("Running batch generation...")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    masks = mask_gen.generate(images[0])
    print("Single generated", len(masks))
    
    try:
        batch_masks = mask_gen.generate(images)
        print("Batch generated length:", len(batch_masks))
    except Exception as e:
        print("Batch generation failed:", e)

print("Done")
