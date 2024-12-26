import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import io
import wandb

def animate_video_with_rewards(frames, rewards, fps=10):
    """
    Create an animation where the left side shows video frames and the right side shows rewards,
    and return an in-memory GIF buffer to log directly to WandB without saving to disk.

    Parameters:
    - frames: numpy array of shape [n, H, W, 3] representing video frames
    - rewards: List or numpy array of rewards, where rewards[i] corresponds to frames[i]
    - fps: Frames per second for the animation
    Returns:
    - gif_buffer: In-memory buffer of the GIF video
    """
    n = len(frames)  # Number of frames

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Left part for video frame
    image_plot = ax1.imshow(frames[0])
    ax1.set_title('Video Frames')
    ax1.axis('off')

    # Right part for rewards
    ax2.set_title('Rewards')
    ax2.set_xlim(0, n - 1)
    ax2.set_ylim(min(rewards) - 1, max(rewards) + 1)
    line_plot, = ax2.plot([], [], lw=2, color='blue')
    scat = ax2.scatter([], [], color='red', zorder=5)

    # Initialize the plot
    def init():
        line_plot.set_data([], [])
        scat.set_offsets(np.empty((0, 2)))  # Ensure it's a 2D array with shape (0, 2)
        return image_plot, line_plot, scat

    # Update function for animation
    def update(frame_idx):
        # Update video frame on the left
        image_plot.set_array(frames[frame_idx])

        # Update rewards plot on the right
        line_plot.set_data(np.arange(frame_idx + 1), rewards[:frame_idx + 1])
        scat.set_offsets(np.array([[frame_idx, rewards[frame_idx]]]))

        return image_plot, line_plot, scat

    # Create animation
    # ani = FuncAnimation(fig, update, frames=n, init_func=init, blit=True, interval=1000//fps)

    # Save the animation frames to an in-memory GIF
    gif_buffer = io.BytesIO()

    # Collect frames to create a GIF
    images = []
    for frame_idx in range(n):
        update(frame_idx)  # Manually update the frame
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(Image.fromarray(img_array))

    # Save the images as a GIF in the buffer
    images[0].save(gif_buffer, format="GIF", save_all=True, append_images=images[1:], duration=1000//fps, loop=0)
    gif_buffer.seek(0)  # Reset buffer to the beginning

    # Close the figure to prevent it from displaying
    plt.close(fig)

    return gif_buffer

# Example usage:
# frames = [...]  # Replace with actual frames, list of numpy arrays (HxWx3)
# rewards = [...]  # Replace with actual rewards, list or numpy array of rewards
# gif_buffer = animate_video_with_rewards(frames, rewards)

# Log the GIF buffer to WandB directly
def log_gif_to_wandb(gif_buffer, gif_name="animation.gif"):
    wandb.log({"animation/" + gif_name: wandb.Video(gif_buffer, format="gif")})

# Example of logging to WandB:
# gif_buffer = animate_video_with_rewards(frames, rewards)
# log_gif_to_wandb


def compute_mmrv(gt_index, cos_sim):
        n = len(gt_index)
        total_violation = 0

        for i in range(n):
            for j in range(i + 1, n):
                # 检查预测进度和帧索引的排序是否一致，不一致则计算排名违约
                if (cos_sim[i] < cos_sim[j]) != (gt_index[i] < gt_index[j]):
                    total_violation += abs(cos_sim[i] - cos_sim[j])

        mmrv = total_violation / n
        return mmrv
# for frame in frames:
#             image_embedding = embedding_image(model, processor, frame)

#             image_embedding = normalize_embeddings(image_embedding)
#             cos_sim.append(compute_similarity(text_embeddings, image_embedding).item())

#         cos_sim = np.array(cos_sim)
#         frame_index = np.linspace(1, len(cos_sim), len(cos_sim))
        
#         gt_index = np.linspace(1, len(cos_sim), len(cos_sim))
#         act_index = np.argsort(cos_sim) + 1
#  mmrv = compute_mmrv(gt_index, cos_sim)
#         wandb.log({f"mmrv_eval/{task}": mmrv})
# 这是mmrv的代码和使用上下文，这是单个task的


def animate_video_with_rewards_class(frames, rewards, num_class = 11, fps=10):
    """
    Create an animation where the left side shows video frames and the right side shows rewards,
    and return an in-memory GIF buffer to log directly to WandB without saving to disk.

    Parameters:
    - frames: numpy array of shape [n, H, W, 3] representing video frames
    - rewards: List or numpy array of rewards, where rewards[i] corresponds to frames[i]
    - fps: Frames per second for the animation
    Returns:
    - gif_buffer: In-memory buffer of the GIF video
    """
    n = len(frames)  # Number of frames

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Left part for video frame
    image_plot = ax1.imshow(frames[0])
    ax1.set_title('Video Frames')
    ax1.axis('off')

    # Right part for rewards
    ax2.set_title('Rewards')
    ax2.set_xlim(0, n - 1)
    ax2.set_ylim(- 1, num_class + 1)
    line_plot, = ax2.plot([], [], lw=2, color='blue')
    scat = ax2.scatter([], [], color='red', zorder=5)

    # Initialize the plot
    def init():
        line_plot.set_data([], [])
        scat.set_offsets(np.empty((0, 2)))  # Ensure it's a 2D array with shape (0, 2)
        return image_plot, line_plot, scat

    # Update function for animation
    def update(frame_idx):
        # Update video frame on the left
        image_plot.set_array(frames[frame_idx])

        # Update rewards plot on the right
        line_plot.set_data(np.arange(frame_idx + 1), rewards[:frame_idx + 1])
        scat.set_offsets(np.array([[frame_idx, rewards[frame_idx]]]))

        return image_plot, line_plot, scat

    # Create animation
    # ani = FuncAnimation(fig, update, frames=n, init_func=init, blit=True, interval=1000//fps)

    # Save the animation frames to an in-memory GIF
    gif_buffer = io.BytesIO()

    # Collect frames to create a GIF
    images = []
    for frame_idx in range(n):
        update(frame_idx)  # Manually update the frame
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(Image.fromarray(img_array))

    # Save the images as a GIF in the buffer
    images[0].save(gif_buffer, format="GIF", save_all=True, append_images=images[1:], duration=1000//fps, loop=0)
    gif_buffer.seek(0)  # Reset buffer to the beginning

    # Close the figure to prevent it from displaying
    plt.close(fig)

    return gif_buffer