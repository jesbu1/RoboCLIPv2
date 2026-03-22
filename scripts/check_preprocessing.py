"""
Check image preprocessing consistency across all stages:
  Stage 1: Reward model training (rewind/utils/processing_utils.py)
  Stage 2a: Online wrapper - dino_load_image only (ideal path)
  Stage 2b: Online wrapper - FULL path through BaseEncoder.encode_images → _encode_image_batch
  Stage 3: generate_labeled_dataset.py (AFTER fix)

This script renders a MetaWorld image and processes it through each pipeline,
then compares the resulting tensors and DINO embeddings.
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Load DINO model ───
print("Loading DINO model...")
dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", force_reload=False).to(device)
dino_model.eval()

# ─── Generate a sample MetaWorld image (640x480) ───
print("\nGenerating sample MetaWorld image...")
import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["button-press-v2-goal-observable"](seed=42)
env.reset()
for _ in range(20):
    env.step(env.action_space.sample())
raw_image = env.sim.render(640, 480, mode="offscreen", camera_name="corner2")[:, :, ::-1]
print(f"Raw image shape: {raw_image.shape}, dtype: {raw_image.dtype}, range: [{raw_image.min()}, {raw_image.max()}]")


# ═══════════════════════════════════════════════════
# Stage 1: Reward model training preprocessing
# ═══════════════════════════════════════════════════
def stage1_preprocess(img_raw):
    """Reward model training: numpy center crop then dino_load_image."""
    h, w = img_raw.shape[:2]
    x = (w - 224) // 2
    y = (h - 224) // 2
    img_cropped = img_raw[y:y+224, x:x+224]

    transform = T.Compose([
        T.ToTensor(),
        T.CenterCrop(224),
        T.Normalize([0.5], [0.5]),
    ])
    img_pil = Image.fromarray(img_cropped)
    tensor = transform(img_pil)[:3].unsqueeze(0)
    return tensor


# ═══════════════════════════════════════════════════
# Stage 2a: Online wrapper - dino_load_image ONLY (bypassing BaseEncoder)
# ═══════════════════════════════════════════════════
def stage2a_preprocess(img_raw):
    """Direct dino_load_image call (what we THOUGHT Stage 2 does)."""
    img_uint8 = img_raw.astype(np.uint8) if img_raw.dtype != np.uint8 else img_raw
    transform = T.Compose([
        T.ToTensor(),
        T.CenterCrop(224),
        T.Normalize([0.5], [0.5]),
    ])
    img_pil = Image.fromarray(img_uint8)
    tensor = transform(img_pil)[:3].unsqueeze(0)
    return tensor


# ═══════════════════════════════════════════════════
# Stage 2b: Online wrapper - FULL path through BaseEncoder → _encode_image_batch
# This is what ACTUALLY happens during online training
# ═══════════════════════════════════════════════════
def stage2b_preprocess_full_path(img_raw):
    """Simulate the EXACT code path in online training."""
    # Step 1: wrapper does image_for_model = image[None, None, :, :, :]
    image_for_model = img_raw[None, None, :, :, :]  # (1, 1, 480, 640, 3) uint8

    # Step 2: BaseEncoder.encode_images
    images = image_for_model
    # transpose HWC → CHW
    if images.shape[-1] == 3 and not images.shape[2] == 3:
        images = np.transpose(images, (0, 1, 4, 2, 3))  # (1, 1, 3, 480, 640) uint8

    # convert to float32 tensor (THIS IS WHERE THE PROBLEM MIGHT BE)
    batch_images = torch.tensor(images[0:1], dtype=torch.float32)  # (1, 1, 3, 480, 640) float32

    # Step 3: _encode_image_batch
    images_np = batch_images.cpu().numpy()  # float32, values [0, 255]

    if images_np.shape[2] == 3:
        images_np = np.transpose(images_np, (0, 1, 3, 4, 2)).squeeze(0)  # (1, 480, 640, 3) float32

    # THE SUSPICIOUS PART: dtype check and conversion
    print(f"\n  [Stage 2b debug] Before uint8 conversion:")
    print(f"    dtype: {images_np.dtype}, range: [{images_np.min():.1f}, {images_np.max():.1f}]")

    if images_np.dtype != np.uint8:
        if images_np.dtype == np.float32 or images_np.dtype == np.float64:
            images_converted = (images_np * 255).astype(np.uint8)
        else:
            images_converted = images_np.astype(np.uint8)
    else:
        images_converted = images_np

    print(f"  [Stage 2b debug] After uint8 conversion:")
    print(f"    dtype: {images_converted.dtype}, range: [{images_converted.min()}, {images_converted.max()}]")

    # Check for overflow
    expected_direct_cast = images_np.astype(np.uint8)
    overflow_pixels = np.sum(images_converted != expected_direct_cast)
    total_pixels = images_converted.size
    print(f"  [Stage 2b debug] Pixels with overflow: {overflow_pixels}/{total_pixels} ({100*overflow_pixels/total_pixels:.1f}%)")

    # Now run dino_load_image on the (possibly corrupted) image
    img_for_dino = images_converted[0]  # (480, 640, 3) uint8
    transform = T.Compose([
        T.ToTensor(),
        T.CenterCrop(224),
        T.Normalize([0.5], [0.5]),
    ])
    img_pil = Image.fromarray(img_for_dino)
    tensor = transform(img_pil)[:3].unsqueeze(0)
    return tensor


# ═══════════════════════════════════════════════════
# Stage 3: generate_labeled_dataset.py (AFTER fix)
# ═══════════════════════════════════════════════════
def stage3_fixed_preprocess(img_raw):
    """Fixed generate_labeled_dataset.py preprocessing."""
    transform = T.Compose([
        T.ToTensor(),
        T.CenterCrop(224),
        T.Normalize([0.5], [0.5]),
    ])
    img_pil = Image.fromarray(img_raw)
    tensor = transform(img_pil)[:3].unsqueeze(0)
    return tensor


# ═══════════════════════════════════════════════════
# Run all pipelines
# ═══════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PREPROCESSING COMPARISON")
print("=" * 70)

tensor1 = stage1_preprocess(raw_image)
tensor2a = stage2a_preprocess(raw_image)
tensor2b = stage2b_preprocess_full_path(raw_image)
tensor3 = stage3_fixed_preprocess(raw_image)

stages = {
    "Stage 1 (reward model training)": tensor1,
    "Stage 2a (wrapper dino_load_image only)": tensor2a,
    "Stage 2b (wrapper FULL BaseEncoder path)": tensor2b,
    "Stage 3 (label script, FIXED)": tensor3,
}

for name, tensor in stages.items():
    print(f"\n{name}:")
    print(f"  Tensor shape: {tensor.shape}")
    print(f"  Tensor range: [{tensor.min():.4f}, {tensor.max():.4f}]")
    print(f"  Tensor mean:  {tensor.mean():.4f}")
    print(f"  Tensor std:   {tensor.std():.4f}")

# ═══════════════════════════════════════════════════
# Compare all pairs
# ═══════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TENSOR COMPARISON (before DINO)")
print("=" * 70)

pairs = [
    ("Stage1", "Stage2a", tensor1, tensor2a),
    ("Stage1", "Stage2b_FULL", tensor1, tensor2b),
    ("Stage2a", "Stage2b_FULL", tensor2a, tensor2b),
    ("Stage1", "Stage3_fixed", tensor1, tensor3),
    ("Stage2b_FULL", "Stage3_fixed", tensor2b, tensor3),
]

for name_a, name_b, ta, tb in pairs:
    diff = (ta - tb).abs()
    status = "IDENTICAL ✓" if diff.max() < 1e-5 else "DIFFERENT ✗"
    print(f"\n{name_a} vs {name_b}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f} → {status}")

# ═══════════════════════════════════════════════════
# Compare DINO embeddings
# ═══════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DINO EMBEDDING COMPARISON")
print("=" * 70)

with torch.no_grad():
    emb1 = dino_model(tensor1.to(device)).cpu()
    emb2a = dino_model(tensor2a.to(device)).cpu()
    emb2b = dino_model(tensor2b.to(device)).cpu()
    emb3 = dino_model(tensor3.to(device)).cpu()

embeddings = {
    "Stage1": emb1,
    "Stage2a": emb2a,
    "Stage2b_FULL": emb2b,
    "Stage3_fixed": emb3,
}

print("\nCosine similarity:")
emb_names = list(embeddings.keys())
for i in range(len(emb_names)):
    for j in range(i+1, len(emb_names)):
        cos = torch.nn.functional.cosine_similarity(
            embeddings[emb_names[i]], embeddings[emb_names[j]]
        ).item()
        l2 = (embeddings[emb_names[i]] - embeddings[emb_names[j]]).norm().item()
        print(f"  {emb_names[i]} vs {emb_names[j]}: cosine={cos:.6f}, L2={l2:.4f}")

# ═══════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════
cos_1_2b = torch.nn.functional.cosine_similarity(emb1, emb2b).item()
cos_2b_3 = torch.nn.functional.cosine_similarity(emb2b, emb3).item()
cos_1_3 = torch.nn.functional.cosine_similarity(emb1, emb3).item()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Key question: Does the FULL online path (Stage 2b) match reward model training (Stage 1)?

  Stage1 vs Stage2b_FULL cosine: {cos_1_2b:.4f}
  Stage2b_FULL vs Stage3_fixed cosine: {cos_2b_3:.4f}
  Stage1 vs Stage3_fixed cosine: {cos_1_3:.4f}
""")

if cos_1_2b < 0.95:
    print("⚠️  CRITICAL: Stage 2b (actual online path) does NOT match Stage 1!")
    print("   The BaseEncoder.encode_images → _encode_image_batch path corrupts")
    print("   the image via float32→uint8 overflow (values [0,255] * 255 → overflow).")
    print("   This means online DINO embeddings are WRONG during training.")
    print("   FIX: Fix _encode_image_batch to handle [0,255] float32 correctly.")
elif cos_1_2b >= 0.95 and cos_1_3 >= 0.95:
    print("✓  All stages are consistent. No preprocessing issues found.")
else:
    print("⚠️  Some stages are inconsistent. Check the numbers above.")
