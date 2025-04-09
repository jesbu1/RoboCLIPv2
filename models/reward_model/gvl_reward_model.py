import torch
import numpy as np
from models.reward_model.base_reward_model import BaseRewardModel
import requests
import base64
import cv2
import json
import os
import random
import re
from typing import Dict, List, Optional


class GVLRewardModel(BaseRewardModel):
    def __init__(
        self,
        api_key: str = "",
        device: str = "cuda",
        max_frames: int = 15,
        offset: float = 0.5,
        batch_size: int = 64,
        success_bonus: float = 50.0,
        reward_at_every_step: bool = False
    ):
        super().__init__(device, batch_size=batch_size, success_bonus=success_bonus)
        self.api_key = api_key
        self.max_frames = max_frames
        self.offset = offset
        self.reward_at_every_step = reward_at_every_step

    def _encode_text_batch(self, text: List[str]) -> np.ndarray: 
        pass

    def _encode_video_batch(self, video_frames: np.ndarray) -> np.ndarray:
        pass

    def encode_text(self, text):
        return super().encode_text(text)
    
    def encode_images(self, images):
        return super().encode_images(images)

    def _calculate_reward_batch(self, video_frames: np.ndarray, text: str) -> np.ndarray:
        pass

    def _encode_image_batch(self, images: torch.Tensor) -> np.ndarray:
        pass

    def extract_frames_from_video(self, frames_array: np.ndarray) -> List[Dict]:
        """
        提取 `max_frames` 个均匀分布的视频帧，并转换成 base64 编码，保留 `gt_index`。
        """
        total_frames = frames_array.shape[0]
        if total_frames == 0:
            print("[!] frames_array is empty, couldn't sample frames。")
            return []

        if total_frames <= self.max_frames:
            temp_indices = list(range(total_frames))  # 直接保留所有帧
        else:
            temp_indices = [0, total_frames - 1]  # 确保首尾帧
            frame_interval = (total_frames - 2) / (self.max_frames - 2)
            temp_indices += [int(1 + self.offset + i * frame_interval) for i in range(self.max_frames - 2)]
            temp_indices = sorted(set(temp_indices))  # 去重+排序
        
        print(f"[INFO] Extracted {temp_indices} frame。")

        frames_info = []
        for idx in temp_indices:
            frame = frames_array[idx]  # (H, W, 3)
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            frames_info.append({
                "gt_index": len(frames_info) + 1, 
                "base64": frame_b64
            })

        return frames_info

    def shuffle_frames(self, frames_info: List[Dict]) -> None:
        """
        随机打乱 frames_info，同时保留 `gt_index` 。
        """
        indices = list(range(1, len(frames_info) + 1))
        random.shuffle(indices)
        for frame, new_idx in zip(frames_info, indices):
            frame["shuffled_index"] = new_idx

    def build_prompt_parts(self, frames_info: List[Dict], task_description: str) -> List[Dict]:
        """
        构建 Gemini API `prompt`，用于推理 `task_completion_percentage`。
        """
        initial_frame = frames_info[0]
        prompt1 = (
            f"You are an expert roboticist tasked to predict task completion percentages "
            f"for frames of a robot for the task of {task_description}. "
            f"The task completion percentages are between 0 and 100, where 100 corresponds to full task completion. "
            f"Note that these frames are in random order, so please pay attention to the individual frames. "
            f"\nInitial robot scene:\nThis frame:"
        )

        prompt2 = (
            f" shows the initial robot scene, where the task completion percentage is 0.\n\n"
            f"Now, for the task of *{task_description}*, output the task completion percentage "
            f"for the following frames that are presented in random order. "
            f"Format your response in JSON as follows, making sure to include all frames:\n\n"
            f"[\n"
            f'  {{"frame_number": i, "frame_description": "...", "task_completion_percentage": 0-100}}\n'
            f"]\n"
        )

        parts = [{"text": prompt1}, {"inline_data": {"mime_type": "image/jpeg", "data": initial_frame["base64"]}}, {"text": prompt2}]

        frames_sorted_by_shuffle = sorted(frames_info, key=lambda f: f["shuffled_index"])
        for i, frame in enumerate(frames_sorted_by_shuffle, start=1):
            parts.append({"text": f"Frame {i}:"})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": frame["base64"]}})

        return parts

    def stream_inference(self, parts: List[Dict]) -> str:
        """
        发送请求到 Gemini API，流式获取文本输出。
        """
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?alt=sse&key={self.api_key}"
        body = {"contents": [{"parts": parts}]}
        headers = {"Content-Type": "application/json"}

        full_text = ""
        with requests.post(url, headers=headers, json=body, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    data_str = line[len("data: "):]
                    if data_str == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data_str)
                        if "candidates" in data_json:
                            candidates = data_json["candidates"]
                            if candidates and len(candidates) > 0:
                                text_piece = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                                full_text += text_piece
                    except json.JSONDecodeError:
                        continue
        return full_text

    @staticmethod
    def extract_json_from_response(text: str) -> str:
        """
        从 Gemini 输出提取 JSON。
        """
        code_block_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")
        match = code_block_pattern.search(text)
        if match:
            return match.group(1).strip()

        array_pattern = re.compile(r"\[\s*\{[\s\S]*?\}\s*\]")
        match = array_pattern.search(text)
        if match:
            return match.group(0).strip()

        return ""

    @staticmethod
    def parse_model_output(model_text: str) -> Optional[List[Dict]]:
        """
        解析 Gemini 返回的 JSON 并转换为 Python 对象。
        """
        json_str = GVLRewardModel.extract_json_from_response(model_text)
        if not json_str:
            return None
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
            return None
        except (json.JSONDecodeError, TypeError):
            return None

    def calculate_rewards(self, video_frames: np.ndarray, text: str) -> float:
        """
        计算 `video_frames` 的 `task_completion_percentage`，返回 `gt_index` 最大的值作为 `reward`。
        """
        frames_info = self.extract_frames_from_video(video_frames)
        self.shuffle_frames(frames_info)
        parts = self.build_prompt_parts(frames_info, text)
        model_output_text = self.stream_inference(parts)
        result_data = self.parse_model_output(model_output_text)
        print(f"[INFO] Model output: {result_data}")

        if result_data:
            # 重新映射回 `gt_index`
            mapped_by_shuffled = {f["shuffled_index"]: f for f in frames_info}
            for item in result_data:
                shuffled_index = item.get("frame_number")
                if shuffled_index in mapped_by_shuffled:
                    mapped_by_shuffled[shuffled_index]["task_completion_percentage"] = item["task_completion_percentage"]

            # 获取 `gt_index` 最大的 `task_completion_percentage`
            sorted_by_gt = sorted(frames_info, key=lambda f: f["gt_index"])
            completion_list = [
                f.get("task_completion_percentage", 0.0) for f in sorted_by_gt
            ]
            gt_index_list = [f["gt_index"] for f in sorted_by_gt]
            print(f"[INFO] GT index list: {gt_index_list}")
            index_list = [f.get("shuffled_index", 0) for f in sorted_by_gt]
            print(f"[INFO] Shuffled index list: {index_list}")
            print(f"[INFO] Task completion list: {completion_list}")
            return sorted_by_gt[-1].get("task_completion_percentage", 0.0)
        

        print("[!] 未能解析有效 JSON，返回默认 Reward 0.")
        return 0.0
    

    @property
    def name(self) -> str:
        """
        Returns the name of the encoder class.
        """
        return 'GVLRewardModel'

