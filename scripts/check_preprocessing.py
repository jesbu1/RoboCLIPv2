"""
Check image preprocessing consistency across 3 stages:
  Stage 1: Reward model training (rewind/utils/processing_utils.py)
  Stage 2: Online wrapper (rewind_no-action-chunk/models/encoders/dino_miniLM_encoder.py)
  Stage 3: generate_labeled_dataset.py

This script renders a MetaWorld image and processes it through each pipeline,
then compares the resulting tensors and DINO embeddings.
"""

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
# Take a few steps so the scene is non-trivial
for _ in range(20):
    env.step(env.action_space.sample())
raw_image = env.render()  # default MetaWorld render
print(f"Raw image shape: {raw_image.shape}, dtype: {raw_image.dtype}, range: [{raw_image.min()}, {raw_image.max()}]")


# ═══════════════════════════════════════════════════
# Stage 1: Reward model training preprocessing
# From: rewind/utils/processing_utils.py
# Note: input is ALREADY center-cropped to 224x224 by metaworld_center_crop.py
# ═══════════════════════════════════════════════════
def stage1_preprocess(img_raw):
    """Simulate the full Stage 1 pipeline."""
    # Step 1: metaworld_center_crop.py does numpy center crop to 224x224
    h, w = img_raw.shape[:2]
    x = (w - 224) // 2
    y = (h - 224) // 2
    img_cropped = img_raw[y:y+224, x:x+224]

    # Step 2: dino_load_image from processing_utils.py
    transform = T.Compose([
        T.ToTensor(),
        T.CenterCrop(224),  # no-op since already 224x224
        T.Normalize([0.5], [0.5]),
    ])
    img_pil = Image.fromarray(img_cropped)
    tensor = transform(img_pil)[:3].unsqueeze(0)
    return tensor, img_cropped


# ═══════════════════════════════════════════════════
# Stage 2: Online wrapper preprocessing
# From: rewind_no-action-chunk/models/encoders/dino_miniLM_encoder.py
# Input is raw 640x480 render
# ═══════════════════════════════════════════════════
def stage2_preprocess(img_raw):
    """Simulate the Stage 2 pipeline."""
    # Wrapper does: image = self.env.render() → (H, W, 3)
    # Then: image_for_model = image[None, None, :, :, :] → (1,1,H,W,3)
    # Then: _encode_image_batch transposes to (1,H,W,3), converts to uint8
    # Then: dino_load_image

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
# Stage 3: generate_labeled_dataset.py preprocessing
# Input is raw image from metaworld_generation.h5
# ═══════════════════════════════════════════════════
def stage3_preprocess(img_raw):
    """Simulate the Stage 3 pipeline."""
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img_raw).unsqueeze(0)
    return tensor


# ═══════════════════════════════════════════════════
# Run all three pipelines
# ═══════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PREPROCESSING COMPARISON")
print("=" * 70)

tensor1, cropped1 = stage1_preprocess(raw_image)
tensor2 = stage2_preprocess(raw_image)
tensor3 = stage3_preprocess(raw_image)

print(f"\nStage 1 (reward model training):")
print(f"  Transform: ToTensor → CenterCrop(224) → Normalize([0.5],[0.5])")
print(f"  Input: numpy center-cropped 224x224 first, then transform")
print(f"  Tensor shape: {tensor1.shape}")
print(f"  Tensor range: [{tensor1.min():.4f}, {tensor1.max():.4f}]")
print(f"  Tensor mean:  {tensor1.mean():.4f}")
print(f"  Tensor std:   {tensor1.std():.4f}")

print(f"\nStage 2 (online wrapper):")
print(f"  Transform: ToTensor → CenterCrop(224) → Normalize([0.5],[0.5])")
print(f"  Input: raw {raw_image.shape[1]}x{raw_image.shape[0]} render")
print(f"  Tensor shape: {tensor2.shape}")
print(f"  Tensor range: [{tensor2.min():.4f}, {tensor2.max():.4f}]")
print(f"  Tensor mean:  {tensor2.mean():.4f}")
print(f"  Tensor std:   {tensor2.std():.4f}")

print(f"\nStage 3 (generate_labeled_dataset.py):")
print(f"  Transform: ToPILImage → Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet)")
print(f"  Input: raw {raw_image.shape[1]}x{raw_image.shape[0]} render")
print(f"  Tensor shape: {tensor3.shape}")
print(f"  Tensor range: [{tensor3.min():.4f}, {tensor3.max():.4f}]")
print(f"  Tensor mean:  {tensor3.mean():.4f}")
print(f"  Tensor std:   {tensor3.std():.4f}")

# ═══════════════════════════════════════════════════
# Compare tensors directly
# ═══════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TENSOR COMPARISON (before DINO)")
print("=" * 70)

diff_12 = (tensor1 - tensor2).abs()
diff_13 = (tensor1 - tensor3).abs()
diff_23 = (tensor2 - tensor3).abs()

print(f"\nStage1 vs Stage2: max_diff={diff_12.max():.6f}, mean_diff={diff_12.mean():.6f}")
print(f"Stage1 vs Stage3: max_diff={diff_13.max():.6f}, mean_diff={diff_13.mean():.6f}")
print(f"Stage2 vs Stage3: max_diff={diff_23.max():.6f}, mean_diff={diff_23.mean():.6f}")

if diff_12.max() < 1e-5:
    print("  → Stage 1 and 2 are IDENTICAL ✓")
else:
    print("  → Stage 1 and 2 are DIFFERENT ✗")

if diff_13.max() < 1e-5:
    print("  → Stage 1 and 3 are IDENTICAL ✓")
else:
    print("  → Stage 1 and 3 are DIFFERENT ✗")

if diff_23.max() < 1e-5:
    print("  → Stage 2 and 3 are IDENTICAL ✓")
else:
    print("  → Stage 2 and 3 are DIFFERENT ✗")

# ═══════════════════════════════════════════════════
# Compare DINO embeddings
# ═══════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DINO EMBEDDING COMPARISON")
print("=" * 70)

with torch.no_grad():
    emb1 = dino_model(tensor1.to(device)).cpu()
    emb2 = dino_model(tensor2.to(device)).cpu()
    emb3 = dino_model(tensor3.to(device)).cpu()

print(f"\nEmbedding shapes: {emb1.shape}, {emb2.shape}, {emb3.shape}")

# Cosine similarity
cos_12 = torch.nn.functional.cosine_similarity(emb1, emb2).item()
cos_13 = torch.nn.functional.cosine_similarity(emb1, emb3).item()
cos_23 = torch.nn.functional.cosine_similarity(emb2, emb3).item()

print(f"\nCosine similarity:")
print(f"  Stage1 vs Stage2: {cos_12:.6f}")
print(f"  Stage1 vs Stage3: {cos_13:.6f}")
print(f"  Stage2 vs Stage3: {cos_23:.6f}")

# L2 distance
l2_12 = (emb1 - emb2).norm().item()
l2_13 = (emb1 - emb3).norm().item()
l2_23 = (emb2 - emb3).norm().item()

print(f"\nL2 distance:")
print(f"  Stage1 vs Stage2: {l2_12:.6f}")
print(f"  Stage1 vs Stage3: {l2_13:.6f}")
print(f"  Stage2 vs Stage3: {l2_23:.6f}")

# ═══════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Stage 1 (reward model training):
  Normalize: [0.5], [0.5]
  Crop: numpy center_crop(224) on 480x640, then T.CenterCrop(224) (no-op)

Stage 2 (online wrapper):
  Normalize: [0.5], [0.5]
  Crop: T.CenterCrop(224) on 480x640 PIL image

Stage 3 (generate_labeled_dataset.py):
  Normalize: ImageNet mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
  Crop: Resize(256) then CenterCrop(224)

Key differences found:
  1. Normalization: Stage 1&2 use [0.5]/[0.5], Stage 3 uses ImageNet stats
  2. Crop pipeline: Stage 3 has Resize(256) before CenterCrop(224)
     - Stage 1&2 see a NARROW center region of the original 640x480
     - Stage 3 sees almost the FULL scene (resized to 256x341, then cropped to 224x224)

Cosine similarity (1.0 = identical, lower = more different):
  Stage1 vs Stage2 (should match): {cos_12:.4f}
  Stage1 vs Stage3 (offline labels): {cos_13:.4f}
  Stage2 vs Stage3 (online vs offline): {cos_23:.4f}
""")

if cos_23 < 0.95:
    print("⚠️  WARNING: Stage 2 (online) and Stage 3 (offline labels) embeddings")
    print("   are significantly different! This means offline and online rewards")
    print("   are computed from inconsistent inputs. This WILL hurt performance.")
    print("   FIX: align generate_labeled_dataset.py preprocessing with the wrapper.")
else:
    print("✓  Stage 2 and 3 embeddings are similar enough.")
