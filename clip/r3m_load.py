

from r3m import load_r3m
import imageio
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.nn as nn
import torch.nn.functional as F


class LangEncoder(nn.Module):
  def __init__(self, device, finetune = False, scratch=False):
    super().__init__()
    
    self.device = device
    self.modelname = "distilbert-base-uncased"
    self.tokenizer = AutoTokenizer.from_pretrained(self.modelname)
    self.model = AutoModel.from_pretrained(self.modelname).to(self.device)
    self.lang_size = 768
      
  def forward(self, langs):
    try:
      langs = langs.tolist()
    except:
      pass
    
    with torch.no_grad():
      encoded_input = self.tokenizer(langs, return_tensors='pt', padding=True)
      input_ids = encoded_input['input_ids'].to(self.device)
      attention_mask = encoded_input['attention_mask'].to(self.device)
      lang_embedding = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
      lang_embedding = lang_embedding.mean(1)
    return lang_embedding





r3m = load_r3m("resnet50") # resnet18, resnet34
r3m.eval().cuda()


video_path = f"/home/jzhang96/RoboCLIPv2/losses/reward_eval_videos/windowclose/GT/1.gif"
frames = imageio.mimread(video_path)
frames = [frame[:,:,:3] for frame in frames]

## DEFINE PREPROCESSING
transforms = T.Compose([T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()]) # ToTensor() divides by 255



language = "Robot closing a door"
lang_encoder = LangEncoder("cuda")
lang_embedding = lang_encoder([language])
import pdb ; pdb.set_trace()

## ENCODE IMAGE
for frame in frames:
    frame = transforms(Image.fromarray(frame.astype(np.uint8))).reshape(-1, 3, 224, 224).cuda()
    with torch.no_grad():
        embedding = r3m(frame * 255.0)
        import pdb ; pdb.set_trace()
        a = 0




print(r3m)