import torch
import re
import h5py
import json
from transformers import pipeline
import os 
huggingface_token = os.environ.get("HF_TOKEN", None)  
#pip install sentencepiece for mistral
#assert huggingface_token is not None, "Please set the HF_TOKEN environment variable to your Hugging Face API token and make sure you were granted access to https://huggingface.co/google/gemma-2-2b-it"
assert huggingface_token is not None, "Please set the HF_TOKEN environment variable to your Hugging Face API token and make sure you were granted access to https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3"

h5_path = "usc_koch_rewind_reward.h5"

# load the language instructions from the h5 path; they're indexed by text strings
def load_lang_instructions(h5_path):
    lang_instructions = {}
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            lang_instructions[key] = [key]
    return lang_instructions

lang_instructions = load_lang_instructions(h5_path)
print(lang_instructions)



pipe = pipeline(
    "text-generation",
    #model="google/gemma-2-2b-it",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    #model="meta-llama/Llama-3.2-3B-Instruct",
    #model="Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device="cuda",  # replace with "mps" to run on a Mac device
    #use_auth_token=True,
    token=huggingface_token,
    do_sample=True,
)

# now, we can generate additional language instructions using the model
for lang_key in lang_instructions:
    while len(lang_instructions[lang_key]) < 10:
        try:
            prompt = f"Please write out 5 additional instructions with the same meaning for the following instruction: {lang_key}. Output the answer in a python list like ```python\n['instr_1', 'instr_2', 'instr_3', 'instr_4']\n``` \n ```python\n"
            outputs = pipe(prompt, max_new_tokens=80)
            response = outputs[0]["generated_text"]
            # remove the prompt from the response
            response = response.replace(prompt, "```python\n")

            # parse the response by looking for the first block within ```python\n...``` using regex
            parsed_response = re.findall(r"```python\n(.*?)```", response, re.DOTALL)[0]

            # remove the new lines from anywhere in the response 
            parsed_response = re.sub(r"\n", "", parsed_response)
            # convert the response to a python list
            exec(f"parsed_response = {parsed_response}")
            # now add unique instructions to the lang_instructions
            for instr in parsed_response:
                # strip the last period
                stripped_instr = instr.strip().strip(".").capitalize()
                if stripped_instr not in lang_instructions[lang_key]:
                    lang_instructions[lang_key].append(stripped_instr)
        except Exception as e:
            print(e)

        print(lang_instructions[lang_key])

# save the instructions as a json
with open(f"additional_lang_instructions_{h5_path}.json", "w") as f:
    json.dump(lang_instructions, f, indent=4)