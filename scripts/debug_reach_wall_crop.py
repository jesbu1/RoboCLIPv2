"""
Debug script: visualize what reach-wall images look like under different preprocessing.
Saves comparison images to figures/ directory.
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ─── Generate reach-wall image ───
print("Setting up reach-wall-v2 environment...")
import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["reach-wall-v2-goal-observable"](seed=42)
env.reset()
for _ in range(30):
    env.step(env.action_space.sample())
raw_image = env.sim.render(640, 480, mode="offscreen", camera_name="corner2")
print(f"Raw image shape: {raw_image.shape}, dtype: {raw_image.dtype}")

os.makedirs("figures", exist_ok=True)


# ─── 1. Original raw image (no preprocessing) ───
raw_pil = Image.fromarray(raw_image)

# ─── 2. Label fix only: Resize(256) → CenterCrop(224) (OLD label script) ───
old_label_transform = T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.CenterCrop(224),
])
old_label_img = old_label_transform(raw_image)

# ─── 3. Fixed label: CenterCrop(224) directly (NEW label script, matches reward model training) ───
new_label_img = T.CenterCrop(224)(raw_pil)

# ─── 4. Full fix with uint8 overflow (OLD _encode_image_batch) ───
# Simulate: uint8 → float32 → *255 → uint8 overflow → CenterCrop(224)
images_float = raw_image.astype(np.float32)  # [0, 255] float32
images_overflow = (images_float * 255).astype(np.uint8)  # overflow!
overflow_pil = Image.fromarray(images_overflow)
overflow_crop_img = T.CenterCrop(224)(overflow_pil)

# ─── 5. Full fix: correct uint8 cast → CenterCrop(224) (NEW _encode_image_batch) ───
images_correct = images_float.astype(np.uint8)  # correct cast
correct_pil = Image.fromarray(images_correct)
correct_crop_img = T.CenterCrop(224)(correct_pil)


# ─── Plot comparison ───
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Full images and crops
axes[0, 0].imshow(raw_pil)
axes[0, 0].set_title(f"1. Raw image (640x480)", fontsize=12)
axes[0, 0].axis('off')
# Draw center crop rectangle
h, w = 480, 640
x1, y1 = (w - 224) // 2, (h - 224) // 2
rect = plt.Rectangle((x1, y1), 224, 224, linewidth=2, edgecolor='red', facecolor='none')
axes[0, 0].add_patch(rect)
axes[0, 0].text(x1, y1 - 5, 'CenterCrop(224)', color='red', fontsize=10)

axes[0, 1].imshow(old_label_img)
axes[0, 1].set_title(f"2. OLD label: Resize(256)→CenterCrop(224)\n(wide view, almost full scene)", fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(new_label_img)
axes[0, 2].set_title(f"3. FIXED label: CenterCrop(224) directly\n(narrow center crop)", fontsize=11)
axes[0, 2].axis('off')

# Row 2: Encoder comparison
axes[1, 0].imshow(overflow_pil)
axes[1, 0].set_title(f"4. OLD encoder: uint8 overflow\n(corrupted full image)", fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(overflow_crop_img)
axes[1, 1].set_title(f"5. OLD encoder: overflow → CenterCrop(224)\n(corrupted, narrow crop)", fontsize=11)
axes[1, 1].axis('off')

axes[1, 2].imshow(correct_crop_img)
axes[1, 2].set_title(f"6. FIXED encoder: correct → CenterCrop(224)\n(correct, narrow crop)", fontsize=11)
axes[1, 2].axis('off')

plt.suptitle("reach-wall-v2: Preprocessing Comparison", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("figures/reach_wall_preprocessing_debug.png", dpi=150, bbox_inches='tight')
print("Saved: figures/reach_wall_preprocessing_debug.png")
plt.close()


# ─── Also compare other tasks to see if reach-wall is special ───
tasks = ["reach-wall-v2", "handle-press-side-v2", "door-lock-v2", "window-close-v2"]
fig, axes = plt.subplots(len(tasks), 3, figsize=(15, 5 * len(tasks)))

for row, task_name in enumerate(tasks):
    env_name = f"{task_name}-goal-observable"
    env2 = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](seed=42)
    env2.reset()
    for _ in range(30):
        env2.step(env2.action_space.sample())
    img = env2.sim.render(640, 480, mode="offscreen", camera_name="corner2")

    img_pil = Image.fromarray(img)

    # Raw with crop rectangle
    axes[row, 0].imshow(img_pil)
    axes[row, 0].set_title(f"{task_name}\nRaw 640x480", fontsize=11)
    axes[row, 0].axis('off')
    h, w = 480, 640
    x1, y1 = (w - 224) // 2, (h - 224) // 2
    rect = plt.Rectangle((x1, y1), 224, 224, linewidth=2, edgecolor='red', facecolor='none')
    axes[row, 0].add_patch(rect)

    # Old label (Resize+CenterCrop) - wide view
    old_img = T.Compose([T.ToPILImage(), T.Resize(256), T.CenterCrop(224)])(img)
    axes[row, 1].imshow(old_img)
    axes[row, 1].set_title(f"OLD: Resize(256)→Crop(224)\n(wide view)", fontsize=11)
    axes[row, 1].axis('off')

    # New label (CenterCrop only) - narrow view
    new_img = T.CenterCrop(224)(img_pil)
    axes[row, 2].imshow(new_img)
    axes[row, 2].set_title(f"FIXED: CenterCrop(224)\n(narrow view)", fontsize=11)
    axes[row, 2].axis('off')

plt.suptitle("Multi-task crop comparison: OLD (wide) vs FIXED (narrow)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("figures/multi_task_crop_comparison.png", dpi=150, bbox_inches='tight')
print("Saved: figures/multi_task_crop_comparison.png")
plt.close()

print("\nDone! Check figures/ directory for the images.")
