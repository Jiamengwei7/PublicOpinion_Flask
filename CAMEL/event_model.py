import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss,TransformerEncoderLayer
from transformers import BertModel,AutoTokenizer
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from allennlp.modules.attention.cosine_attention import CosineAttention
from allennlp.nn.util import weighted_sum
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPConfig, CLIPTokenizerFast
from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths
from allennlp.nn.util import masked_softmax
from allennlp.nn import util
import numpy as np
from collections import OrderedDict


from transformers import BertTokenizer
bert_dir = '/media/dell/XuTing/PublicOpinion_Flask/CAMEL/model/bert-base-cased'
clip_dir = '/media/dell/XuTing/PublicOpinion_Flask/CAMEL/model/clip-vit-base-pacth16'
LABELS_list= ['None','Conflict:Attack', 'Conflict:Demonstrate', 'Contact:Meet', 'Contact:Phone-Write', 'Justice:Arrest-Jail', 'Life:Die', 'Movement:Transport', 'Transaction:Transfer-Money']
device = "cuda" if torch.cuda.is_available() else "cpu"
class Unify_model_new(nn.Module):
    def __init__(self, bert_dir, clip_dir):
        print("Loading model")
        super().__init__()

        self.y_num1 = 9
        self.y_num2 = 9

        # init Bert
        self.bert = BertModel.from_pretrained(bert_dir, output_hidden_states=True)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_dir,max_length=512,truncation=True, padding="max_length",do_lower_case=False)
        self.span_extractor = EndpointSpanExtractor(input_dim=self.bert.config.hidden_size, combination='x')
        self.T_sim_proj = nn.Linear(512, self.bert.config.hidden_size)
        self.V_sim_proj = nn.Linear(512, self.bert.config.hidden_size)
        self.bert_fc = nn.Linear(self.bert.config.hidden_size + 768, self.y_num1)
        print('resume bert checkpoint from %s' % bert_dir)

        # init CLIP
        self.clip = CLIPModel.from_pretrained(clip_dir)
        self.clip_tokenizer = CLIPTokenizerFast.from_pretrained(clip_dir)
        self.clip_fc = nn.Linear(self.bert.config.hidden_size + 768, self.y_num2)
        print('resume clip checkpoint from %s' % clip_dir)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(512, 512 // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(512 // 16, 512)),
            ("layernorm1", nn.LayerNorm(512)),
        ]))
        self.meta = False

        self.attention_block = AttentionBlock(n_head=12, d_model=768, d_k=64, d_v=64, d_in=768, d_hid=2048, dropout=0.1)


    def getName(self):
        return self.__class__.__name__

    def forward(self, type, input, device):
        if type=='ace':
            loss, _ = self.forward1(input,device)
        else:   # 传入了（’imSitu‘, imSitu_batch, device）
            loss, _ = self.forward2(input, device, True, V_LS, V_Mixup)  # 紧接着就调用了forward2函数
        return loss

    # token-level sim
    def compute_logits(self, data_x, bert_mask, data_span, image):
        # torch.Size([10, 5, 512]) image

        outputs = self.bert(data_x, attention_mask=bert_mask)
        hidden_states = outputs[2]

        # torch.Size([10, 181, 768])
        temp = torch.cat(hidden_states[-1:], dim=-1)
        # remove cls token # torch.Size([10, 180, 768])reshape(10,180,768)
        temp = temp[:,1:,:].contiguous()
        temp = self.span_extractor(temp, data_span)   # temp(10,180,768)

        # torch.Size([10, 5, 768])
        # if self.meta:
        #     meta_output = self.meta_net(image)
        #     image = meta_output + image
        image = image.to(device)
        image = image.float()
        sim_img = self.T_sim_proj(image)  ##目前是（10，3，768）
        # new_vector = torch.zeros((10, 177, 768)).cuda()
        # sim_img = torch.cat([sim_img, new_vector], dim=1)

        # With attention block
        # ttt = self.T_attention_block(temp,sim_img,sim_img,False).squeeze(1)
        # print("temp", temp.shape)
        # print("sim_img", sim_img.shape)

        ttt = self.attention_block(temp, sim_img, sim_img, False).squeeze(1)

        # torch.Size([10, 180, 1280])
        temp = torch.cat([temp, ttt], dim=-1)
        logits = self.bert_fc(temp)
        return logits


    def forward1(self, input, device):

        # data_x, bert_mask, data_span, sequence_mask, image, text, words= input
        data_x = input['data_x']
        bert_mask = input['bert_mask']
        data_span = input['data_span_tensor']
        sequence_mask = input['sequence_mask']
        image = input['images']
        text = input['text']
        # print("input", input)
        # data_x, bert_mask, data_span, sequence_mask, image = data_x.to(device), bert_mask.to(
        #     device), data_span.to(device), sequence_mask.to(device), image.to(device)
        print("type(data_x):", type(data_x))
        print(data_x.shape)
        data_x = data_x.to(device)
        
        bert_mask = bert_mask.to(device)
        data_span = data_span.to(device)
        sequence_mask = sequence_mask.to(device)
        print("image:", image.shape)
        # img_feature = self.clip.get_image_features(pixel_values=image)
        img_feature = image  # ([1, 2, 512])
        batch_size = 1
        input_img_feat = img_feature.view(batch_size, 2, 512)  # 应该变成（10, 5, 512）

        # add cls token CLS标记的输出包含了整个句子的语义信息，对于句子级别的分类任务非常有用。
        # 创建一个形状为（batch_size,）的整数张量t1，其中每个元素都是101
        t1 = torch.tensor([101]*batch_size, device=device).long()
        t2 = torch.tensor([1] * batch_size, device=device).long()
        # 将t1张量的形状从(batch_size, )调整为(batch_size, 1)
        t1 = t1.reshape(batch_size, 1)  #（10，1）
        t2 = t2.reshape(batch_size, 1)
        data_x = torch.cat([t1,data_x], dim=1)  # 变成了（10，181）
        print("data_x:",data_x.shape)
        print("t2:",t2.shape)
        print("bert_mask:", bert_mask.shape)
        # 将t2张量的形状从(batch_size, )调整为(batch_size, 1)
        # t2 = t2.unsqueeze(1)
        print("t2:",t2.shape)
        # 去掉bert-mask中间的维度
        bert_mask = bert_mask.squeeze(1)
        print("改后的bert_mask:", bert_mask.shape)
        bert_mask = torch.cat([t2, bert_mask], dim=1) # 变成了（10，181）

        logits = self.compute_logits(data_x, bert_mask, data_span, input_img_feat)

        return logits
    # Image attend to Text
    # forward2(input, device, M2E2, LS=0.0, Mixup=0.0)
    def forward2(self, input, device, M2E2, LS=0.0, Mixup=0.0):
        images, data_y, sentences = input[0].to(device),input[1].to(device),input[2]
        # Mixup:通过将输入数据和标签进行混合，生成新的训练样本，从而提升模型的泛化能力
        if Mixup>0:
            images, data_y, y_mix, lam = mixup_data(images, data_y, Mixup, device)

        img_feature = self.clip.get_image_features(pixel_values=images) #(64,512)
        batch_size = img_feature.shape[0]
        # print("batch_size:",batch_size)

        if M2E2:
            length = list()
            flag = 0
            sentence = list()
            # for sen in sentences:
            #     # print(len(sentence))
            #     if sen is not None:
            #         length.append([flag, flag + len(sen)])
            #         sentence.extend(sen)
            #         flag += len(sen)
            for sen in sentences:
                # if isinstance(sen, list):
                #     sen = sen[0]
                print(len(sentence))
                length.append([flag, flag + len(sen)])
                sentence.extend(sen)
                flag += len(sen)
            # inputs是
            # print("sentence:", len(sentence))
            inputs = self.bert_tokenizer(sentence, padding=True, return_tensors="pt",truncation=True).to(device)
            # print("inputs:", inputs.size())  # torch.Size([64, 51])
            last_hidden_state = self.bert(**inputs).last_hidden_state # last_hidden_state: torch.Size([64, 51, 768])
            # print("last_hidden_state:", last_hidden_state.shape) #torch.Size([640, 51, 768])
            txt_feature = last_hidden_state[:, 0, :].squeeze(1)
            # print("after:txt_feature = last_hidden_state[:, 0, :].squeeze(1):",txt_feature.shape) # torch.Size([640, 768])
            for i, txt_index in enumerate(length):
                txt = torch.unsqueeze(txt_feature[txt_index[0]:txt_index[1]], dim=0)
                if i == 0:
                    input_txt_feat = txt
                else:
                    # print("txt", txt.shape)
                    # print("input_txt_feat:", input_txt_feat.shape)
                    input_txt_feat = torch.cat((input_txt_feat, txt), 0)
            # print("input_txt_feat:",input_txt_feat.shape) # torch.Size([10, 64, 768])
            # print("meta", self.meta)  # False
            if self.meta:
                meta_output = self.meta_net(img_feature)
                img = meta_output + img_feature
            else:  # 执行的是这里
                img = img_feature
            sim_img = self.V_sim_proj(img)
            # print( "sim_img = self.V_sim_proj(img):",sim_img.shape)   # torch.Size([64, 768])
            # input_txt_feat: torch.Size([64, 10, 768])
            # With attention block
            # cap = self.V_attention_block(sim_img.unsqueeze(1), input_txt_feat, input_txt_feat, True).squeeze(1)
            input_txt_feat = input_txt_feat.permute(1, 0, 2)
            input_txt_feat = input_txt_feat[:, 0, :].squeeze(1)
            input_txt_feat = input_txt_feat.unsqueeze(1)
            # print("after:input_txt_feat = input_txt_feat.view(64, 1, 768):",input_txt_feat.shape)
            cap = self.attention_block(sim_img.unsqueeze(1), input_txt_feat, input_txt_feat, True).squeeze(1)
            # cap = self.attention_block(sim_img.unsqueeze(1).expand(-1, 10, -1), input_txt_feat, input_txt_feat,True).squeeze(1) # sim_img: torch.Size([64, 1, 768]) input_txt_feat: torch.Size([64, 10, 768]) input_txt_feat: torch.Size([64, 10, 768])
            # fc_input = torch.cat([cap, sim_img], dim=-1)
            fc_input = torch.cat([cap, sim_img], dim=-1)  # sim_img: torch.Size([64, 768]), cap: torch.Size([64, 10,768])
        else:
            # add cls with meta 768+512
            # print(len(sentences))
            inputs = self.bert_tokenizer(sentences, padding=True, return_tensors="pt").to(device)
            # print("执行一次")
            last_hidden_state = self.bert(**inputs).last_hidden_state
            cap = last_hidden_state[:, 0, :].squeeze(1)
            # fake_token = self.fake_token.unsqueeze(0).expand(batch_size, -1)
            if self.meta:
                meta_output = self.meta_net(img_feature)
                img = meta_output + img_feature
            else:
                img = img_feature

            sim_img = self.V_sim_proj(img)

            # With attention block
            cap = cap.unsqueeze(0)
            cap = cap.expand(batch_size, cap.shape[1], cap.shape[2])
            # cap = self.V_attention_block(sim_img.unsqueeze(1), cap, cap,True).squeeze(1)
            cap = self.attention_block(sim_img.unsqueeze(1), cap, cap, True).squeeze(1)
            fc_input = torch.cat([cap, sim_img], dim=-1)  # torch.Size([64, 10, 1536])
        # print("cap.size()", cap.size())  # cap.size() torch.Size([64, 10, 768])
        # print("sim_img.size()", sim_img.size())  # torch.Size([64, 768])
        # print("before clip_fc的fc_input.size", fc_input.size())  # torch.Size([64, 10, 1536])
        logits = self.clip_fc(fc_input)  # logits.size() torch.Size([64, 10, 9])  或许应该是([64, 1, 9])

        if LS>0:
            loss_fct = CrossEntropyLoss(label_smoothing=LS)
        else:
            loss_fct = CrossEntropyLoss()

        if Mixup>0:
            loss = lam * loss_fct(logits.view(-1, self.y_num2), data_y.view(-1)) + (1 - lam) * loss_fct(logits.view(-1, self.y_num2), y_mix.view(-1))
        else:
            # print("logits.size()",logits.size())
            # print("data_y.size()",data_y.size())
            #logits.size() torch.Size([64, 10, 9]) data_y.size() torch.Size([64])
            #loss_fct的input形状：(N, C), N是batchsize, C是类别数，所以logits应该是[64,9]或[64,1,9]
            loss = loss_fct(logits.view(-1, self.y_num2), data_y.view(-1))

        return loss, logits


    @torch.no_grad()
    def predict_text_label(self, input, device):
        # sequence_mask = input[3].to(device)
        sequence_mask = input['sequence_mask'].to(device)

        logits = self.forward1(input, device)
        print("logits", logits)

        classifications = torch.argmax(logits, -1)
        # print("classifications", classifications)
        classifications = list(classifications.cpu().numpy())
        predicts = []
        for classification, mask in zip(classifications, sequence_mask):
            predicts.append(classification[:])

        return predicts

    @torch.no_grad()
    # (images, labels, sentences), device, M2E2
  

    @torch.no_grad()
    # indicate the fusion metheod before using this function
    def get_pnorm_logits(self, input, p, M2E2, device):
        images, data_y, sentences = input[0].to(device), input[1].to(device), input[2]
        img_feature = self.clip.get_image_features(pixel_values=images)
        batch_size = img_feature.shape[0]

        if M2E2:
            length = list()
            flag = 0
            sentence = list()
            for sen in sentences:
                length.append([flag, flag + len(sen)])
                sentence.extend(sen)
                flag += len(sen)

            inputs = self.bert_tokenizer(sentence, padding=True, return_tensors="pt").to(device)
            last_hidden_state = self.bert(**inputs).last_hidden_state
            txt_feature = last_hidden_state[:, 0, :].squeeze(1)
            for i, txt_index in enumerate(length):
                txt = torch.unsqueeze(txt_feature[txt_index[0]:txt_index[1]], dim=0)
                if i == 0:
                    input_txt_feat = txt
                else:
                    input_txt_feat = torch.cat((input_txt_feat, txt), 0)

            if self.meta:
                meta_output = self.meta_net(img_feature)
                img = meta_output + img_feature
            else:
                img = img_feature
            sim_img = self.V_sim_proj(img)

            # With attention block
            cap = self.attention_block(sim_img.unsqueeze(1), input_txt_feat, input_txt_feat, True).squeeze(1)

            # fc_input = torch.cat([cap, img], dim=-1)
            fc_input = torch.cat([cap, sim_img], dim=-1)
        else:

            # add cls with meta 768+512
            inputs = self.bert_tokenizer(sentences, padding=True, return_tensors="pt").to(device)
            last_hidden_state = self.bert(**inputs).last_hidden_state
            cap = last_hidden_state[:, 0, :].squeeze(1)
            # fake_token = self.fake_token.unsqueeze(0).expand(batch_size, -1)
            if self.meta:
                meta_output = self.meta_net(img_feature)
                img = meta_output + img_feature
            else:
                img = img_feature

            sim_img = self.V_sim_proj(img)

            # With attention block
            cap = cap.unsqueeze(0)
            cap = cap.expand(batch_size, cap.shape[1], cap.shape[2])
            # cap = self.V_attention_block(sim_img.unsqueeze(1), cap, cap,True).squeeze(1)
            cap = self.attention_block(sim_img.unsqueeze(1), cap, cap, True).squeeze(1)

            # fc_input = torch.cat([cap, img], dim=-1)
            fc_input = torch.cat([cap, sim_img], dim=-1)

        def pnorm(weights, p):
            normB = torch.norm(weights, 2, 1)
            ws = weights.clone()
            for i in range(weights.size(0)):
                ws[i] = ws[i] / torch.pow(normB[i], p)
            return ws

        ws = pnorm(self.clip_fc.state_dict()['weight'], p)

        logits = torch.mm(fc_input, ws.t())
        return logits

    # Get fusion feature
    def get_fusion_feature(self, images, sentences, device, M2E2):

        img_feature = self.clip.get_image_features(pixel_values=images)
        img_feature_concatenate = self.clip.vision_model(pixel_values=images).last_hidden_state
        batch_size = img_feature.shape[0]

        if M2E2:
            length = list()
            flag = 0
            sentence = list()
            for sen in sentences:
                length.append([flag, flag + len(sen)])
                sentence.extend(sen)
                flag += len(sen)

            inputs = self.bert_tokenizer(sentence, padding=True, return_tensors="pt").to(device)
            last_hidden_state = self.bert(**inputs).last_hidden_state
            txt_feature = last_hidden_state[:, 0, :].squeeze(1)
            for i, txt_index in enumerate(length):
                txt = torch.unsqueeze(txt_feature[txt_index[0]:txt_index[1]], dim=0)
                if i == 0:
                    input_txt_feat = txt
                else:
                    input_txt_feat = torch.cat((input_txt_feat, txt), 0)

            sim_img = self.V_sim_proj(img_feature)

            # With attention block
            cap = self.attention_block(sim_img.unsqueeze(1), input_txt_feat, input_txt_feat, True)
        else:

            inputs = self.bert_tokenizer(sentences, padding=True, return_tensors="pt").to(device)
            last_hidden_state = self.bert(**inputs).last_hidden_state
            cap = last_hidden_state[:, 0, :].squeeze(1)

            sim_img = self.V_sim_proj(img_feature)

            # With attention block
            cap = cap.unsqueeze(0)
            cap = cap.expand(batch_size, cap.shape[1], cap.shape[2])
            cap = self.attention_block(sim_img.unsqueeze(1), cap, cap, True)

        fusion_feature = torch.cat([cap, sim_img.unsqueeze(1), img_feature_concatenate[:, 1:,:]], dim=1)

        return fusion_feature


# 实现了缩放点积注意力的核心计算逻辑
# 常用于Transformer模型的自注意力机制中，用于对输入序列进行建模和特征提取
# 通过设置温度参数，可以在一定程度上控制注意力权重的范围和分布

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

# imgtxt
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.cls = nn.Linear(d_model, n_head * d_k, bias=False)
        self.token = nn.Linear(d_model, n_head * d_k, bias=False)
        self.img = nn.Linear(d_model, n_head * d_v, bias=False)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, flag, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        # print(len_k)
        # print(len_v)

        residual = q
        # print("q的形状:",q.shape)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # print(flag)
        if flag:
            # print(self.img(q).shape)
            q = self.img(q).view(sz_b, len_q, n_head, d_k)# sz_b:64  len_q: 1  n_head: 12  d_k: 64
            # print(q.shape)
            # print(self.cls(k).shape)
            # print(self.cls(v).shape)
            k = self.cls(k)
            # print("sz_b:",sz_b)
            k = k.view(sz_b, len_k, n_head, d_k)
            # k = k.view(sz_b, -1, n_head, d_k)
            # sz_b:64  len_k: 64  n_head: 12  d_k: 64 cls(k):torch.Size([10, 64, 768])
            # v = self.cls(v).view(sz_b, len_v, n_head, d_v)  # sz_b:64  len_v: 64  n_head: 12  d_v: 64 cls(v):torch.Size([10, 64, 768])
            v = self.cls(v).view(sz_b, -1, n_head, d_v)
        else:
            q = self.token(q).view(sz_b, len_q, n_head, d_k)
            k = self.img(k).view(sz_b, len_k, n_head, d_k)
            v = self.img(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
# 两层前馈神经网络，通过非线性变换和残差链接来处理输入特征，并使用层归一化进行归一化处理
# Dropout是一种常用的正则化技术，旨在减少深度学习模型的过拟合问题
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.MultiHeadAttention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.PositionwiseFeedForward = PositionwiseFeedForward(d_in, d_hid, dropout)
    # forward() 方法定义了模型的前向传播逻辑
    def forward(self, q, k, v, flag=True):
        output1, attn = self.MultiHeadAttention(q, k, v, flag)
        output = self.PositionwiseFeedForward(output1) # 位置前馈神经网络

        return output
