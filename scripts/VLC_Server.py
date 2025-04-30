import os
import pickle
import numpy as np
import torch
import flask
from flask import Flask, request, jsonify
import io
import base64
from PIL import Image
import argparse

from video_language_critic.reward import RewardCalculator
#from paths import *
from VLC_Reward import SingleVideoVLCRewardCalculator, read_video_as_frames, load_vlc_args 

app = Flask(__name__)

vlc_ckpt_name = "ckpt_mw40_retrank33_tigt_negonly_a_rf_1__pytorch_model.bin.20"
vlc_args = load_vlc_args(vlc_ckpt_name)

reward_calculator = SingleVideoVLCRewardCalculator(
    vlc_args=vlc_args,
    caption_text="Press the button",
    env_id="button-press-v2",
    stretch_partial_videos=True,
    device="cuda"
)

def decode_image(image_data):
    """从 base64 字符串解码为 numpy 数组"""
    image_bytes = base64.b64decode(image_data)
    image_array = np.load(io.BytesIO(image_bytes))
    return image_array

@app.route("/compute_reward", methods=["POST"])
def compute_reward():
    """接收视频帧 (numpy array) + 文本，计算 Reward"""
    request_data = pickle.loads(request.get_data())

    image_data = request_data["video"]
    text = request_data["text"]
    reward_calculator.set_text(text)

    video_frames = decode_image(image_data)

    #Reward
    video_tensor = reward_calculator.transform_frames(video_frames)
    vlm_reward = reward_calculator.compute_video_reward(video_tensor)
    print(f"vlm_reward: {vlm_reward}")
    return jsonify({"reward": vlm_reward})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 5000)))
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port)
    print("Server started!")
