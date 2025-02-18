import json
import h5py
from clip_utils import load_model, embedding_text, get_full_liv_embedding

annotations = json.load(open("../llm_utils/additional_lang_instructions_usc_koch_rewind_reward.h5.json", "r"))

h5_file = h5py.File("usc_koch_rewind_dino_reward_side_main.h5", 'a')

model, processor, tokenizer = load_model("liv")

model = model.to("cuda")


for key in h5_file.keys():
    anns = annotations[key]

    text_embedding = embedding_text(model, tokenizer, anns).detach().cpu().numpy()
    print(text_embedding.shape)
    del h5_file[key]["liv_lang_embedding"]
    h5_file[key].create_dataset("liv_lang_embedding", data=text_embedding)
        # h5_file[key].create_dataset(f"liv_lang_embedding_individual_{i}", data=text_embedding)



for key in h5_file.keys():
    anns = annotations[key]

    for i in range(len(anns)):
        ann = anns[i]
        text_embedding = get_full_liv_embedding(model, processor, ann).detach().cpu().numpy()
        del h5_file[key][f"liv_lang_embedding_individual_{i}"]
        h5_file[key].create_dataset(f"liv_lang_embedding_individual_{i}", data=text_embedding)



# for key in annotations.keys():

#     if key not in h5_file:
#         print("Key not found: ", key)

#     else:

#         text = annotations[key]
#         text_embedding = embedding_text(model, tokenizer, text).detach().cpu().numpy()
#         group.create_dataset("lang_embedding", data=text_embedding)






