# coding:utf-8
import sys
import io
from PIL import Image
import requests
import open_clip
import torch
from transformers import BertModel, BertConfig, BertTokenizer
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import pickle
import shutil
sys.stdout = io.TextIOWrapper(sys.stdout.detach(),encoding='utf-8')

# 加载Taiyi 中文 text encoder
text_tokenizer = BertTokenizer.from_pretrained("/media/dell/ChenXue/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese")
text_encoder = BertModel.from_pretrained("/media/dell/ChenXue/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese").eval()

# 加载openclip的image encoder
clip_model, _, processor = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
clip_model = clip_model.eval()

def mostsim_Image(query_text,images_features,imgs_ids):
  text = text_tokenizer(query_text, return_tensors='pt', padding=True)['input_ids']
  text_features = text_encoder(text)[1]
  text_features = text_features / text_features.norm(dim=1, keepdim=True)
  sim_matrix = []
  with torch.no_grad():
    for image_feature in images_features:
      image_feature = image_feature / image_feature.norm(dim=1, keepdim=True)
      logits_per_image = image_feature @ text_features.t()
      sim_matrix.append(logits_per_image.cpu().numpy().tolist()[0][0])
  sort_sim = sorted(sim_matrix, reverse = True)
  # print(sort_sim)
  # max_sim = max(sim_matrix)
  # max_sim = sort_sim[0]
  # max_index = sim_matrix.index(max_sim)
  # mostsim_img = imgs_ids[max_index]
  top_5 = sort_sim[:9]
  for img_score in top_5:
    img_index = sim_matrix.index(img_score)
    img_path = imgs_ids[img_index]
    print(img_path,img_score)
    img_path_name = img_path.split('/')[-1]
    new_path = 'resources/result1'+'/'+img_path_name
    #img_path = '/media/dell/ChenXue/'+img_path
    shutil.copy(img_path,new_path)

  # return mostsim_img

def main():
    # embedding_cache_path = 'resources/img.pkl'
    embedding_cache_path = 'img.pkl'
    # print("Load pre-computed features from disc")
    imgs_ids, images_features = [], []
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        imgs_ids = cache_data['img_ids']
        images_features = cache_data['features']
    img_path = mostsim_Image(sys.argv[1],images_features,imgs_ids)
    # print(img_path)
if __name__ == "__main__":
	main()
# print(sys.argv[1])
