import json
import h5py
from clip_utils import load_model, embedding_text

annotations = json.load(open("../llm_utils/additional_lang_instructions_usc_koch_rewind_reward.h5.json", "r"))

h5_file = h5py.File("usc_koch_rewind_reward_concat.h5", 'a')

model, processor, tokenizer = load_model("liv")

model = model.to("cuda")

for key in annotations.keys():
    if key not in h5_file:
        print("Key not found: ", key)

    else:
        group = h5_file[key]
        if "lang_embedding" in group:
            del group["lang_embedding"]
        text = annotations[key]
        text_embedding = embedding_text(model, tokenizer, text).detach().cpu().numpy()
        group.create_dataset("lang_embedding", data=text_embedding)






