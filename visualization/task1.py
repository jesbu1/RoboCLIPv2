import PIL
from s3dg import S3D
import torch as th
import json
#from kitchen_env_wrappers import readGif
import numpy as np
import os
import cv2


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

def find_label(video_filename, json_path='labels/train.json'):
    """
    Finds the corresponding label for a given video filename in the train.json file.

    Parameters:
    - video_filename: Name of the video file (without the extension).
    - json_path: Path to the 'train.json' file.

    Returns:
    - The found label as a string, or None if not found.
    """
    try:
        # Open and load the JSON file
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)

        for item in data:
            if item['id'] == video_filename:
                return item['label']

        return None

    except FileNotFoundError:
        print(f"file '{json_path}' Not Found")
        return None
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

def adjust_frames(frames):
    """
    Ensures an even number of frames. If there's an odd number of frames, removes the last frame.
    """
    if frames.shape[2] % 2 != 0:  # Check the T dimension
        frames = frames[:, :, :-1, :, :]  # Remove the last frame
    return frames

def Embedding(model, video_path = None, text_label = None):
    """
    Generates embeddings for video and text using a given model.

    Parameters:
    - model: The model used for generating embeddings.
    - video_path: Path to the video file.
    - text_label: Text label corresponding to the video.

    Returns:
    - video_embedding, text_embedding: Embeddings for the video and text.
    """
    if text_label:
        text_output = model.text_module([text_label])
        text_embedding = text_output['text_embedding']
    if video_path:
        video_id = get_filename_without_extension(video_path)
        text_label = find_label(video_id)
        print(text_label)
        frames = read_webm_frames(video_path)
        frames = preprocess_human_demo(frames)
        frames = adjust_frames(frames)
        if frames.shape[1] > 3:
            frames = frames[:, :3]
        video = th.from_numpy(frames)
        print(video.shape)

        video_output = model(video.float())
        video_embedding = video_output['video_embedding']
        text_output = model.text_module([text_label])
        text_embedding = text_output['text_embedding']
        return video_embedding, text_embedding

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

    target_height = 240
    target_width = 320

    height, width, _ = frames[0].shape
    start_x = width // 2 - target_width // 2
    start_y = height // 2 - target_height // 2

    cropped_frames = [
        frame[start_y:start_y + target_height, start_x:start_x + target_width]
        for frame in frames
    ]

    return np.array(cropped_frames)


# test_frame = readGif('gifs/dense_original.gif')
# test_frame = preprocess_human_demo(test_frame)
# video = th.from_numpy(test_frame)
# print(test_frame.shape)
s3d = S3D('./s3d_dict.npy', 512)
s3d.load_state_dict(th.load('./s3d_howto100m.pth'))
s3d.eval()
video_embedding, text_embedding = Embedding(s3d, './20bn-something-something-v2/7.webm')
#print(video_embedding.shape, text_embedding.shape)
l2_distances = th.norm(text_embedding - video_embedding, p=2, dim=1)

mean_distance = th.mean(l2_distances)
std_distance = th.std(l2_distances)

print("Mean L2 Distance:", mean_distance.item())
print("STD L2 Distance:", std_distance.item())

