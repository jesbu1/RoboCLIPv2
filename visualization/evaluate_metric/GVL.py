import cv2
import base64
import random
import math
import re
import requests
import json
from typing import List, Dict, Optional
import numpy as np  # 确保已经安装了 numpy


class GeminiVideoAnalyzerHDF5:
    def __init__(
        self,
        api_key: str,
        frames_array: np.ndarray,
        task_description: str,
        max_frames: int = 15,
        offset: float = 0.5
    ):
        """
        :param api_key:          Gemini 的 API Key
        :param frames_array:     来自 HDF5 中的 (N, H, W, 3) uint8 数组（视频的逐帧数据）
        :param task_description: 机器人任务描述
        :param max_frames:       如果帧数 N > max_frames，则只抽取 max_frames 个帧
        :param offset:           采样帧时的时间偏移（秒/帧）。原本用于实际秒，如保持和前端一致的含义即可。
        """
        self.api_key = api_key
        # 视频帧序列: shape = (N, H, W, 3)
        self.frames_array = frames_array  
        self.task_description = task_description
        self.max_frames = max_frames
        self.offset = offset

        # 存放帧信息的列表：每个元素包含
        # {"gt_index": i, "shuffled_index": ..., "base64": "..."}
        self.frames_info: List[Dict] = []

    def extract_frames_from_memory(self) -> None:
        """
        改进逻辑：
        - 若 N <= max_frames，保留之前的做法(几乎可以得到首帧与尾帧)；
        - 若 N > max_frames，则：
        1) 强制包含第 0 帧 与 第 N-1 帧
        2) 对中间的 (N-2) 区间做均匀抽样 (max_frames - 2) 帧
        3) 将上面所有索引合并、去重、排序后，再进行编码
        """
        total_frames = self.frames_array.shape[0]
        if total_frames == 0:
            print("[!] frames_array 为空，无法提取帧。")
            self.frames_info = []
            return

        # 如果总帧数 <= max_frames：保留原先逻辑
        if total_frames <= self.max_frames:
            frame_count = total_frames
            frame_interval = 1.0
            print(f"Extracting all {frame_count} frames (<= max_frames).")

            temp_indices = []
            for i in range(frame_count):
                # 原先你用 offset + i*frame_interval 然后 int(...) 做索引：
                sample_time = self.offset + i * frame_interval
                frame_index = int(sample_time)
                if 0 <= frame_index < total_frames:
                    temp_indices.append(frame_index)

        else:
            # total_frames > max_frames
            print(f"Extracting exactly {self.max_frames} frames from total={total_frames}, ensuring first & last included.")
            # 1) 固定收集第 0 帧 & 第 N-1 帧
            temp_indices = [0, total_frames - 1]

            # 2) 均匀抽取 (max_frames - 2) 个中间帧
            #    注：若你仍想保留 offset，可以写成 offset + i*frame_interval
            #    但 offset 对于第一帧=0是否还有意义，需要看你需求。
            #    这里示例保留 offset，让中间帧也带 0.5 偏移。
            inner_count = self.max_frames - 2
            if (total_frames - 2) <= 0:
                # 如果只有1帧或2帧，就不做中间抽样了
                print("Warning: total_frames - 2 <= 0, can't sample middle frames.")
            else:
                frame_interval = (total_frames - 2) / float(inner_count)
                for i in range(inner_count):
                    sample_time = self.offset + i * frame_interval
                    # 中间帧索引落在 [1, N-2]
                    frame_index = int(1 + sample_time)  
                    if 1 <= frame_index < (total_frames - 1):
                        temp_indices.append(frame_index)

            # 去重并排序
            temp_indices = sorted(set(temp_indices))
            print(f"Extracted {len(temp_indices)} frames: {temp_indices}")

        # ============ 将索引列表转成 JPEG + base64 ============
        temp_frames_info = []
        for idx in temp_indices:
            frame = self.frames_array[idx]  # shape = (H, W, 3)
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            temp_frames_info.append({
                "gt_index": len(temp_frames_info) + 1,
                "base64": frame_b64
            })

        self.frames_info = temp_frames_info


    def shuffle_frames(self) -> None:
        """
        随机打乱 frames_info。类似原先 shuffle_frames。
        """
        indices = list(range(1, len(self.frames_info) + 1))
        random.shuffle(indices)
        for frame, new_idx in zip(self.frames_info, indices):
            frame["shuffled_index"] = new_idx

    def build_prompt_parts(self) -> List[Dict]:
        """
        构建请求时的 `parts` 列表，遵循之前的逻辑：
          - 首先找 gt_index=1 的帧作“初始场景”
          - 添加 prompt1 + 这帧
          - 添加 prompt2
          - 再按 shuffled_index 排序依次插入 “Frame i: ” + 该帧
        """
        # 找到 gt_index = 1 的帧
        initial_frame = next((f for f in self.frames_info if f["gt_index"] == 1), None)
        if not initial_frame:
            # 若没有(极端情况), 就用第一个
            if not self.frames_info:
                print("[!] 无可用帧，无法构建 prompt_parts")
                return []
            initial_frame = self.frames_info[0]

        prompt1 = (
            f"You are an expert roboticist tasked to predict task completion percentages "
            f"for frames of a robot for the task of {self.task_description}. "
            f"The task completion percentages are between 0 and 100, where 100 corresponds to full task completion. "
            f"Note that these frames are in random order, so please pay attention to the individual frames. "
            f"\nInitial robot scene:\nThis frame:"
        )

        prompt2 = (
            f" shows the initial robot scene, where the task completion percentage is 0.\n\n"
            f"Now, for the task of *{self.task_description}*, output the task completion percentage "
            f"for the following frames that are presented in random order. "
            f"Format your response in JSON as follows, making sure to include all frames:\n\n"
            f"[\n"
            f'  {{"frame_number": i, "frame_description": "...", "task_completion_percentage": 0-100}}\n'
            f"]\n"
        )

        parts = []
        # 1) prompt1
        parts.append({"text": prompt1})
        # 2) initial_frame inline
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": initial_frame["base64"]
            }
        })
        # 3) prompt2
        parts.append({"text": prompt2})

        # 4) “Frame X” + inline, 按 shuffled_index 排序
        frames_sorted_by_shuffle = sorted(self.frames_info, key=lambda f: f["shuffled_index"])

        for i, frame in enumerate(frames_sorted_by_shuffle, start=1):
            parts.append({"text": f"Frame {i}:"})
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": frame["base64"]
                }
            })

        return parts

    def stream_inference(self, parts: List[Dict]) -> str:
        """
        调用 Gemini SSE 接口，返回流式拼接后的完整文本。
        """
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?alt=sse&key={self.api_key}"
        body = {
            "contents": [
                {
                    "parts": parts
                }
            ]
        }
        headers = {
            "Content-Type": "application/json"
        }

        full_text = ""
        with requests.post(url, headers=headers, json=body, stream=True) as resp:
            resp.raise_for_status()

            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[len("data: "):]
                    if data_str == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data_str)
                        candidates = data_json.get("candidates")
                        if candidates and len(candidates) > 0:
                            content = candidates[0].get("content", {})
                            parts_list = content.get("parts", [])
                            if parts_list:
                                text_piece = parts_list[0].get("text", "")
                                full_text += text_piece
                    except json.JSONDecodeError:
                        continue
        return full_text

    @staticmethod
    def extract_json_from_response(text: str) -> str:
        """
        与前端JS类似的从大段文本中提取 JSON 的逻辑：
        - 优先查找 ```json ... ``` 代码块
        - 否则再尝试匹配类似 [... { ... }, ...]
        如果找不到，返回空字符串。
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
        尝试从模型大段文本中提取 JSON，并用 json.loads 转成 Python 对象（列表）。
        如果失败，返回 None。
        """
        json_str = GeminiVideoAnalyzerHDF5.extract_json_from_response(model_text)
        if not json_str:
            return None
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
            return None
        except (json.JSONDecodeError, TypeError):
            return None

    def run_analysis(self) -> List[Optional[float]]:
        """
        核心流程：
          1) 从内存数组提取并编码帧
          2) 随机打乱
          3) 构建 prompt
          4) SSE 推理
          5) 解析 JSON
          6) 根据 gt_index 返回完成度数组

        :return: 与原始帧顺序 (gt_index) 对应的完成度列表
        """
        # 1) 提取帧
        self.extract_frames_from_memory()
        if not self.frames_info:
            print("[!] No frames extracted from memory array.")
            return []

        # 2) 随机打乱
        self.shuffle_frames()

        # 3) 构建 prompt
        parts = self.build_prompt_parts()
        if not parts:
            print("[!] build_prompt_parts 失败，可能没有任何帧数据")
            return []

        # 4) SSE 推理
        model_output_text = self.stream_inference(parts)

        # 5) 解析 JSON
        result_data = self.parse_model_output(model_output_text)
        if result_data is None:
            print("[!] 未能提取到有效 JSON，请检查模型输出。")
            print("Full model output text:", model_output_text)
            return []

        # 6) 把模型输出映射回 gt_index
        mapped_by_shuffled = {}
        for item in result_data:
            sidx = item.get("frame_number")
            if isinstance(sidx, int):
                mapped_by_shuffled[sidx] = item

        # 在 frames_info 里记录模型结果
        for frame in self.frames_info:
            sidx = frame.get("shuffled_index")
            if sidx in mapped_by_shuffled:
                frame["model_output"] = mapped_by_shuffled[sidx]
            else:
                frame["model_output"] = None

        # 按 gt_index 升序遍历
        frames_in_gt_order = sorted(self.frames_info, key=lambda f: f["gt_index"])
        task_completion_list = []
        index_list = []
        gt_index_list = []
        for f in frames_in_gt_order:
            if f["model_output"] is not None:
                task_completion_list.append(f["model_output"].get("task_completion_percentage"))
                index_list.append(f["model_output"].get("frame_number"))
                gt_index_list.append(f["gt_index"])
            else:
                task_completion_list.append(None)
        print("Task completion list:", task_completion_list)
        return task_completion_list #, index_list, gt_index_list


# ============== 示例：如何读取 HDF5 并调用这个类 ==============
if __name__ == "__main__":
    import h5py

    # 假设你的 h5 文件路径
    h5_path = "/scr/yusenluo/RoboCLIP/visualization/decoder_only/metaworld_GT_eval_v2.h5"
    dataset_name = "button-press-v2"  # 例如

    with h5py.File(h5_path, 'r') as f:
        # 取到 (N, 224, 224, 3) 的 uint8 数据
        frames_data = f[dataset_name][:]
        # 取文本描述
        task_bytes = f["text_annotations"][f"{dataset_name}_text"][()]
        # decode bytes to str
        task_text = task_bytes.decode("utf-8")

    # 实例化并调用
    analyzer = GeminiVideoAnalyzerHDF5(
        api_key="AIzaSyCaDj-o-VuadUwA94U9VdirB81VsY_t3TM",
        frames_array=frames_data,
        task_description=task_text,  # or any custom string
    )
    completion_array = analyzer.run_analysis()
    # print("Final completion array:", completion_array)
    # print("Index list:", index_list)
    # print("GT index list:", gt_index_list)
