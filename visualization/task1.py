import PIL
from s3dg import S3D
import torch as th
import json
#from kitchen_env_wrappers import readGif
import numpy as np
import os
import cv2
from pca import plot_embeddings_3d,plot_embeddings

def readGif(filename, asNumpy=True):
    """ readGif(filename, asNumpy=True)

    Read images from an animated GIF file.  Returns a list of numpy
    arrays, or, if asNumpy is false, a list if PIL images.

    """

    # Check PIL
    if PIL is None:
        raise RuntimeError("Need PIL to read animated gif files.")

    # Check Numpy
    if np is None:
        raise RuntimeError("Need Numpy to read animated gif files.")

    # Check whether it exists
    if not os.path.isfile(filename):
        raise IOError('File not found: ' + str(filename))

    # Load file using PIL
    pilIm = PIL.Image.open(filename)
    pilIm.seek(0)

    # Read all images inside
    images = []
    try:
        while True:
            # Get image as numpy array
            tmp = pilIm.convert()  # Make without palette
            a = np.asarray(tmp)
            if len(a.shape) == 0:
                raise MemoryError("Too little memory to convert PIL image to array")
            # Store, and next
            images.append(a)
            pilIm.seek(pilIm.tell() + 1)
    except EOFError:
        pass

    # Convert to normal PIL images if needed
    if not asNumpy:
        images2 = images
        images = []
        for im in images2:
            images.append(PIL.Image.fromarray(im))

    # Done
    return images


def read_webm_frames(video_path):
    """
    Reads frames from a .webm video file using OpenCV.

    Parameters:
    - video_path: Path to the video file.

    Returns:
    - frames: A list of video frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert color space from BGR to RGB since OpenCV uses BGR by default
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames


def find_label(video_filename, json_paths=['../labels/train.json', '../labels/validation.json']):
    """
    Finds the corresponding label for a given video filename in the train.json or validation.json file.

    Parameters:
    - video_filename: Name of the video file (without the extension).
    - json_paths: List of paths to the 'train.json' and 'validation.json' files.

    Returns:
    - The found label as a string, or None if not found.
    """
    for json_path in json_paths:
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r') as json_file:
                    data = json.load(json_file)

                for item in data:
                    if item['id'] == video_filename:
                        return item['label']

        except FileNotFoundError:
            print(f"file '{json_path}' Not Found")
        except json.JSONDecodeError:
            print(f"file '{json_path}' not valid json")

    return None

def get_filename_without_extension(file_path):
    """Returns the filename without the extension for a given file path."""
    file_name_with_extension = os.path.basename(file_path)
    file_name_without_extension, _ = os.path.splitext(file_name_with_extension)
    return file_name_without_extension

def preprocess_human_demo(frames):
    """
    Preprocesses frames for video by adjusting size, adding a batch dimension,
    and transposing dimensions to match S3D model input requirements.

    Parameters:
    - frames: A list of video frames.

    Returns:
    - Preprocessed frames as a numpy array.
    """
    frames = np.array(frames)
    frames = adjust_size(frames)
    frames = frames[None, :,:,:,:]
    frames = frames.transpose(0, 4, 1, 2, 3)
    return frames

def adjust_frames(frames, target_frame_count = 32):
    """
    Ensures same numbers of frames(32).
    """
    _, _, frame_count, _, _ = frames.shape
    frames = th.from_numpy(frames)
    if frame_count < target_frame_count:
        blank_frames = th.zeros(
            (frames.shape[0], frames.shape[1], target_frame_count - frame_count, frames.shape[3], frames.shape[4]),
            dtype=frames.dtype)
        adjusted_frames = th.cat((frames, blank_frames), dim=2)

    elif frame_count > target_frame_count:
        indices = th.linspace(0, frame_count - 1, target_frame_count, dtype=th.long)
        adjusted_frames = th.index_select(frames, 2, indices)
    else:
        adjusted_frames = frames

    return adjusted_frames

def Embedding(model, video_paths = None):
    """
    Generates embeddings for video and text using a given model.

    Parameters:
    - model: The model used for generating embeddings.
    - video_path: Path to the video file.
    - text_label: Text label corresponding to the video.

    Returns:
    - video_embedding, text_embedding: Embeddings for the video and text.
    """

    batch_video = []
    batch_text = []
    if video_paths:
        for video_path in video_paths:
            #print(video_path)
            video_id = get_filename_without_extension(video_path)
            text_label = find_label(video_id)
            # if text_label == None:
            #     print(video_path)
            frames = read_webm_frames(video_path)
            frames = preprocess_human_demo(frames)
            frames = adjust_frames(frames)
            if frames.shape[1] > 3:
                frames = frames[:, :3]
            video = frames
            batch_video.append(video)
            batch_text.append(text_label)

        videos = th.cat(batch_video, dim=0)
        with th.no_grad():
            video_output = model(videos.float())
            video_embeddings = video_output['video_embedding']
            text_output = model.text_module(batch_text)
            text_embeddings = text_output['text_embedding']
            # video_embeddings.append(video_embedding)
            # text_embeddings.append(text_embedding)
            # batched_video_tensor = th.cat(video_embeddings, dim=0)
            # batched_text_tensor = th.cat(text_embeddings, dim=0)
        return video_embeddings, text_embeddings

def adjust_size(frames):
    """
    Adjusts the size of the frames to a target height and width by cropping.

    Parameters:
    - frames: A list of video frames.

    Returns:
    - Cropped frames as a numpy array.
    """
    if len(frames) == 0:
        return np.array([])

    target_height = 224
    target_width = 224

    height, width, _ = frames[0].shape
    start_x = width // 2 - target_width // 2
    start_y = height // 2 - target_height // 2

    cropped_frames = [
        frame[start_y:start_y + target_height, start_x:start_x + target_width]
        for frame in frames
    ]

    return np.array(cropped_frames)

def list_webm_files(folder_path):
    """
    Lists all .webm files within a given folder.

    Parameters:
    - folder_path (str): The path to the folder contains the dataset(webm).

    Returns:
    - list: A list of full paths to the .webm files within the specified folder.
    """
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.webm')]




# test_frame = readGif('gifs/dense_original.gif')
# test_frame = preprocess_human_demo(test_frame)
# video = th.from_numpy(test_frame)
# print(test_frame.shape)
video_paths = list_webm_files('vidz4jesse')
#print(video_paths)
s3d = S3D('../s3d_dict.npy', 512)
s3d.load_state_dict(th.load('../s3d_howto100m.pth'))
s3d.eval()
video_embeddings, text_embeddings = Embedding(s3d, video_paths)
#print(video_embeddings.shape, text_embeddings.shape)
l2_distances = th.norm(text_embeddings - video_embeddings, p=2, dim=1)
similarity_scores = th.matmul(text_embeddings, video_embeddings.t())
#print(l2_distances.shape)
mean_distance = th.mean(l2_distances)
std_distance = th.std(l2_distances)
min_distance = th.min(l2_distances)
max_distance = th.max(l2_distances)

mean_score = th.mean(similarity_scores)
min_score = th.min(similarity_scores)
max_score = th.max(similarity_scores)
std_score = th.std(similarity_scores)

print("Mean similarity score:", mean_score.item())
print("Min similarity score:", min_score.item())
print("Max similarity score:", max_score.item())
print("STD of similarity scores:", std_score.item())
print("Mean L2 Distance:", mean_distance.item())
print("STD L2 Distance:", std_distance.item())
print("Min L2 Distance:", min_distance.item())
print("Max L2 Distance:", max_distance.item())
stats_info = (
    f"Mean similarity score: {mean_score.item():.2f}\n"
    f"Min similarity score: {min_score.item():.2f}\n"
    f"Max similarity score: {max_score.item():.2f}\n"
    f"STD of similarity scores: {std_score.item():.2f}\n"
    f"Mean L2 Distance: {mean_distance.item():.2f}\n"
    f"STD L2 Distance: {std_distance.item():.2f}\n"
    f"Min L2 Distance: {min_distance.item():.2f}\n"
    f"Max L2 Distance: {max_distance.item():.2f}"
)


plot_embeddings(video_embeddings, text_embeddings, directory_name='plots', file_name='pca_plot_4.png')
plot_embeddings_3d(video_embeddings, text_embeddings, directory_name='plots', file_name='pca_plot_3d_4.png')