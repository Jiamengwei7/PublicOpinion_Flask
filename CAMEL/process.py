from flask import Flask,request,jsonify,template_rendered
from flask_cors import CORS
from transformers import BertTokenizer
import torch
from PIL import Image
import torch.nn as nn
import nltk
import clip
import pickle
import os
import time
from tqdm import tqdm
from CAMEL.event_model import Unify_model_new
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
bert_dir = '/media/dell/XuTing/PublicOpinion_Flask/CAMEL/model/bert-base-cased'
from transformers import AutoTokenizer
from CAMEL.event_dataset import ACE_Dataset
tokenizer = AutoTokenizer.from_pretrained(bert_dir, do_lower_case=False)
CKPT = '/media/dell/XuTing/PublicOpinion_Flask/CAMEL/model/model_T_4.pth'
event_type = ['Attack', 'Demonstrate', 'Meet', 'Phone-Write', 'Arrest-Jail', 'Die', 'Transport', 'Transfer-Money']
event_type = ['O'] + event_type #把O加入到列表中
tag2idx = {tag: idx for idx, tag in enumerate(event_type)}
idx2tag = {idx: tag for idx, tag in enumerate(event_type)}
print("tag2idx",tag2idx)
model_clip, preprocess = clip.load("/media/dell/XuTing/PublicOpinion_Flask/CAMEL/model/ViT-B-32.pt", device=device)
def _to_bert_examples(sen, labels):
    subword_ids = list()
    spans = list()
    label_ids = list()

    for word in sen:
        sub_tokens = tokenizer.tokenize(word)
        sub_tokens = tokenizer.convert_tokens_to_ids(sub_tokens)

        s = len(subword_ids)
        subword_ids.extend(sub_tokens)
        e = len(subword_ids) - 1
        # 这个span其实就是子词的范围，比如“new”这个单词没被拆分，那就是[4,4],比如“U.S.”这个词被拆分成“U”,".","S","."这四个子词，那span就是[16,21]
        spans.append([s, e])

    for label in labels:
        label_ids.append(tag2idx.get(label, 0))

    return subword_ids, spans, label_ids

def _build_clip(elem):
    print("tag2idx",tag2idx)
    image, text, = elem[-2:]

    with torch.no_grad():
        text = clip.tokenize(' '.join(text)[:50]).to(device)
        text_features = model_clip.encode_text(text)
        print("process中的mage",image)
        path = 'uploads'
        for i in range(len(image)):  # 对每个图像都进行处理
          #每个图片拼接path
          image[i] = path + '/' + image[i]
        print("修改后的image",image)
        # image = ["D://Study//PublicOpinion_Vue//public//static//image-ourself//war_151_0.jpg"]
        images = [preprocess(Image.open(i)).unsqueeze(0).to(device) for i in image]  # 处理每个图像为张量列表
        images = torch.cat(images, 0)  # 将张量列表连接成一个张量
        image_features = model_clip.encode_image(images)

        elem.insert(-2, text_features.cpu().numpy())
        elem.insert(-2, image_features.cpu().numpy())
        return elem

# def _transfer(l, p, idx2tag):
def _transfer(p, idx2tag):
    # if l == 'O':
    #     l = 'O'
    # else:
    #     l = 'B-' + l.replace('-', '_')
    p = idx2tag[p]
    if p == 'O':
        p = 'O'
    else:
        p = 'B-' + p.replace('-', '_')
    return p
def evaluate_text(model, dataloader, device):   
    filename = "/media/dell/XuTing/PublicOpinion_Flask/CAMEL/predict.conll"
    # fileout = open(filename, 'w')
    all_words = []
    all_labels = []
    all_predicts = []
    total_loss = 0
    for batch in tqdm(dataloader):
        # print("batchaaaaa", batch)
        # print("batch[-2]", batch[-2])
        # print("batch[-3]", batch[-3])
        predicts = model.predict_text_label(batch, device)
        all_predicts.extend(predicts)
        print("predicts", predicts)
        # print("predict中的batch", batch)
        # # words, labels = batch[-3], batch[-2]
        # text = batch['text']
        # nltk.download('punkt') 
        # words = nltk.word_tokenize(text)
        # print("predict中的words", words)
        # all_words.extend(words)
        # 把predicts写进conll文件
        fileout = open(filename, 'w')
        # 逐行写入预测结果到 Conll 文件
        # for predict in  predicts:
        #     # fileout = open(filename, 'w')
        #     line = f"{predict}\n"
        #     fileout.write(line)
        #     # 关闭文件
        #     fileout.close()
        for predicts in all_predicts:
          for p in predicts:
            p = _transfer(p, idx2tag)
            print(p, file=fileout)
          print(file=fileout)
        fileout.close()
        # all_labels.extend(labels)

    # for words, labels, predicts in zip(all_words, all_labels, all_predicts):
    #     for w, l, p in zip(words, labels, predicts):
    #         l, p = _transfer(l, p, idx2tag)
    #         print(w, l, p, file=fileout)
    #     print(file=fileout)
    # fileout.close()

    # with open(filename) as fout:
    #     eval_results = evaluate_conll_file(fout)
    # with open("/media/dell/XuTing/PublicOpinion_Flask/CAMEL/TEE.json", 'w') as f:
    #     json.dump(sen2type_dict, f)

def create_dataset():
    with open("/media/dell/XuTing/PublicOpinion_Flask/CAMEL/data.pkl", "rb") as f:
        m2e2 = pickle.load(f)
    dataset_text = ACE_Dataset(m2e2)
    dataloader_text = DataLoader(dataset_text, batch_size=1)
    return dataloader_text
