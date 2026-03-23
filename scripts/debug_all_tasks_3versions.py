"""
Debug: show all 8 tasks under 3 code versions:
  V1: No fixes (original buggy code) - OLD label + OLD encoder
  V2: Label fix only - FIXED label + OLD encoder (uint8 overflow)
  V3: Both fixes - FIXED label + FIXED encoder
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

print("Setting up environments...")
import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

os.makedirs("figures", exist_ok=True)

TASKS = [
    "window-close-v2",
    "reach-wall-v2",
    "faucet-close-v2",
    "coffee-button-v2",
    "button-press-wall-v2",
    "door-lock-v2",
    "handle-press-side-v2",
    "sweep-into-v2",
]


def render_task(task_name):
    env_name = f"{task_name}-goal-observable"
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](seed=42)
    env.reset()
    for _ in range(30):
        env.step(env.action_space.sample())
    img = env.sim.render(640, 480, mode="offscreen", camera_name="corner2")
    return img


def v1_no_fixes(img_raw):
    """V1: No fixes. OLD label (Resize+Crop) + OLD encoder (uint8 overflow).
    This is what OFFLINE sees (label) + what ONLINE sees (encoder)."""
    # What offline H5 label script saw (OLD preprocessing):
    # Resize(256) → CenterCrop(224)
    label_transform = T.Compose([T.ToPILImage(), T.Resize(256), T.CenterCrop(224)])
    label_img = label_transform(img_raw)

    # What online encoder saw (uint8 overflow → CenterCrop):
    images_float = img_raw.astype(np.float32)
    images_overflow = (images_float * 255).astype(np.uint8)
    overflow_pil = Image.fromarray(images_overflow)
    encoder_img = T.CenterCrop(224)(overflow_pil)

    return label_img, encoder_img


def v2_label_fix_only(img_raw):
    """V2: Label fix only. FIXED label (CenterCrop) + OLD encoder (uint8 overflow)."""
    # What offline H5 label script sees (FIXED):
    label_img = T.CenterCrop(224)(Image.fromarray(img_raw))

    # What online encoder sees (still buggy, uint8 overflow):
    images_float = img_raw.astype(np.float32)
    images_overflow = (images_float * 255).astype(np.uint8)
    overflow_pil = Image.fromarray(images_overflow)
    encoder_img = T.CenterCrop(224)(overflow_pil)

    return label_img, encoder_img


def v3_both_fixes(img_raw):
    """V3: Both fixes. FIXED label (CenterCrop) + FIXED encoder (correct uint8)."""
    # What offline H5 label script sees (FIXED):
    label_img = T.CenterCrop(224)(Image.fromarray(img_raw))

    # What online encoder sees (FIXED):
    encoder_img = T.CenterCrop(224)(Image.fromarray(img_raw))

    return label_img, encoder_img


# ─── Plot all tasks × 3 versions ───
fig, axes = plt.subplots(len(TASKS), 7, figsize=(28, 4 * len(TASKS)))

col_titles = [
    "Raw 640x480",
    "V1 Label\n(Resize→Crop)\nOLD, wide view",
    "V1 Encoder\n(overflow→Crop)\nOLD, colors broken",
    "V2 Label\n(CenterCrop)\nFIXED",
    "V2 Encoder\n(overflow→Crop)\nOLD, colors broken",
    "V3 Label\n(CenterCrop)\nFIXED",
    "V3 Encoder\n(CenterCrop)\nFIXED, correct",
]

for row, task in enumerate(TASKS):
    print(f"Rendering {task}...")
    img = render_task(task)

    v1_label, v1_enc = v1_no_fixes(img)
    v2_label, v2_enc = v2_label_fix_only(img)
    v3_label, v3_enc = v3_both_fixes(img)

    images = [
        Image.fromarray(img),
        v1_label, v1_enc,
        v2_label, v2_enc,
        v3_label, v3_enc,
    ]

    for col, (im, title) in enumerate(zip(images, col_titles)):
        axes[row, col].imshow(im)
        axes[row, col].axis('off')
        if row == 0:
            axes[row, col].set_title(title, fontsize=9, fontweight='bold')
        if col == 0:
            axes[row, col].set_ylabel(task, fontsize=10, fontweight='bold', rotation=0,
                                       labelpad=120, va='center')

    # Draw crop rectangle on raw image
    h, w = 480, 640
    x1, y1 = (w - 224) // 2, (h - 224) // 2
    rect = plt.Rectangle((x1, y1), 224, 224, linewidth=1.5, edgecolor='red', facecolor='none')
    axes[row, 0].add_patch(rect)

plt.suptitle(
    "All 8 tasks: V1 (no fixes) vs V2 (label fix only) vs V3 (both fixes)\n"
    "Label = what offline H5 data sees | Encoder = what online training sees",
    fontsize=14, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig("figures/all_tasks_3versions.png", dpi=120, bbox_inches='tight')
print("\nSaved: figures/all_tasks_3versions.png")
plt.close()
