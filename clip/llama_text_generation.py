import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "I have a task for a robot to solve:'Pressing handle from side' Please give me a bunch of ways to rephrase the task description without changing the meaning. "},
    {"role": "user", "content": "Can you generate the task description in a list format, ie.e save the task description in a list?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
import pdb; pdb.set_trace()
print(outputs[0]["generated_text"][-1])