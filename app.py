import json
from unicodedata import name
from flask import Flask, request, jsonify
# 解决跨域问题
from flask_cors import CORS
import subprocess
import mysql.connector
from oneie.predict_text import predict
import base64
import nltk
import pickle
import torch
import time
from CAMEL.process import _to_bert_examples, _build_clip, evaluate_text, create_dataset
from CAMEL.event_model import Unify_model_new
CKPT = '/media/dell/XuTing/PublicOpinion_Flask/CAMEL/model/model_T_4.pth'
device = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import BartTokenizerFast,BertTokenizer
import torch
import torch.nn as nn
tokenizer=BertTokenizer.from_pretrained('/media/dell/pretrainModel/fnlp/bart-base')
tokenizer.add_special_tokens({'additional_special_tokens': ["<trigger>", "</trigger>"]})
label_id_dict={'Causal':0,'Follow':1,'NoRel':2}
id_label_dict={0:'Causal',1:'Follow',2:'NoRel'}
text_id_dict={'因果':0,'跟随':1,'无关':2}
id_label_text={0:'因果',1:'跟随',2:'无关'}
def label_to_num(label):
    return label_id_dict[label]
def num_to_label(num):
    return id_label_dict[num]

# 创建Flask实例
app = Flask(__name__)
app.config.from_object(__name__)
from model import PERE
# 解决跨域
CORS(app, resources={r'/*': {'origins': '*'}}, supports_credentials=True)

# 连接数据库
# 创建MySQL连接
conn = mysql.connector.connect(
    host='172.20.137.141',
    user='root',
    password='123456',
    database='PublicOpinion'
)
cursor = conn.cursor()

# 搜索接口
@app.route('/api/search', methods=['POST', 'OPTIONS'])
def search():
    if request.method == 'OPTIONS':
        # 处理预检请求
        response_headers = {
            'Access-Control-Allow-Origin': '*',  # 允许所有来源的请求
            'Access-Control-Allow-Methods': 'POST',  # 允许的方法
            'Access-Control-Allow-Headers': 'Content-Type',  # 允许的头部信息
        }
        return '', 200, response_headers

    query = request.json.get('query')
    script_path = 'test.py'  # Python 脚本的路径
    python_path = '/root/anaconda3/envs/opinion/bin/python'  # Python 解释器的路径

    cmd = [python_path, script_path, query]  # 构建执行脚本的命令

    try:
        result = subprocess.check_output(cmd)  # 执行命令并获取输出结果
        result = result.decode('utf-8')  # 将字节流转换为字符串
        return jsonify(result=result)
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e))
    

    # 返回搜索结果作为 JSON 响应
    # return jsonify(query)

# 监听数据集获取端口
@app.route('/api/getDataList', methods=['GET'])
def get_data_list():
  try:
    # 执行SQL查询
    with conn.cursor() as cursor:
        sql = 'SELECT * FROM my_data'
        cursor.execute(sql)
        results = cursor.fetchall()
        # print(json.dumps(results))
        # 将查询结果转换为列表
        payload = []
        content = {}
        for result in results:
            content = {'id': result[0], 'num': result[1], 'content': result[2]}
            payload.append(content)
            content = {}
        return jsonify({'code': 200, 'message': payload})

  except Exception as e:
    return jsonify({'code': 10001, 'message': str(e)})

@app.route('/api/sendEvent',methods=['POST'])
def textEEPredict():
  text=request.json['text']
  language=request.json["language"]
  res=predict(language,text)
  return jsonify({'code': 200, 'result':res})

@app.route('/api/uploadImage', methods=['POST'])
def upload():
    # a = request.files.get('data')
    print(request.files)
    if 'image' in request.files:
        file = request.files['image']
        # 处理上传的文件，保存到指定位置或进行进一步的操作
        # 这里只是简单地将文件保存到当前目录下的 "uploads" 文件夹中
        file.save('uploads/' + file.filename)
        
        # 将文件内容转换为 Base64
        with open('uploads/' + file.filename, 'rb') as f:
            file_content = f.read()
            base64_content = base64.b64encode(file_content).decode('utf-8')

        # 构造返回数据
        response_data = {
            'name': file.filename,
            'dataURL': f'data:image/jpeg;base64,{base64_content}'  # 返回图片的 Base64 数据 URL
        }
        
        # 返回上传成功的响应
        return jsonify(response_data), 200
    else:
        # 返回未选择文件的错误信息
        response_data = {'message': '未选择文件'}
        return jsonify(response_data), 400

@app.route('/api/sendMultiData', methods=['POST'])
def processMultiData():
  # process_path = process.py
  global result
  data = request.json
  nltk.download('punkt')
  print("111", data)
  # 对接收到的数据进行预处理,text为输入的文本
  text = data['text']
  text_token = nltk.word_tokenize(text)
  image = data['image']
  sen = text_token + ['[SEP]']
  labels = ['O'] * len(sen)
  subword_ids, spans, label_ids  = _to_bert_examples(sen, labels)
  result = _build_clip([subword_ids, spans, label_ids, image, text_token])
  # time.sleep(1)  # 延迟1秒
  print("预处理完毕",result)
  with open('/media/dell/XuTing/PublicOpinion_Flask/CAMEL/data.pkl', 'wb') as f:
    pickle.dump(result, f)
  # time.sleep(2)  # 延迟1秒
  # 加载数据
  dataloader_text = create_dataset()
  print("加载数据完毕")
  # 加载模型
  model = Unify_model_new("/media/dell/XuTing/PublicOpinion_Flask/CAMEL/model/bert-base-cased", "/media/dell/XuTing/PublicOpinion_Flask/CAMEL/model/clip-vit-base-pacth16")
  model.load_state_dict(torch.load(CKPT, map_location='cpu'), strict=False)
  model.to(device)
  print('Loaded model from %s' % CKPT)
  # 进行预测
  evaluate_text(model, dataloader_text, device)
  filename = "/media/dell/XuTing/PublicOpinion_Flask/CAMEL/predict.conll"
  non_O_content = []
  with open(filename, 'r') as file:
      for position, line in enumerate(file, start=1):
          line = line.strip()  # 去除行尾的换行符和空白字符
          if line != 'O':
              non_O_content.append((position, line))
  print(non_O_content)
  # time.sleep(3)  # 延迟1秒
  # 执行SQL查询
  with conn.cursor() as cursor:
    # 修改 SQL 查询以同时选择 arguments 和 role 字段
    query = "SELECT arguments, role FROM argument_result WHERE sentence = %s"
    cursor.execute(query, (text,))
    
    # 使用 fetchall() 获取所有结果
    results = cursor.fetchall()

# 打印结果，json.dumps 将结果转换为 JSON 格式的字符串
  print(json.dumps(results, ensure_ascii=False))  # 使用 ensure_ascii=False 以支持中文字符

  arguments_list = []
  roles_list = []

  for result in results:
    # 假设每个 result 是一个元组 (arguments, role)
    arguments, role = result
    arguments_list.append(arguments)
    roles_list.append(role)
      # 使用 jsonify 返回 JSON 响应
  res = {
      'code': 200,
      'message': 'Success',
      'data': {
          'non_O_content': non_O_content,
          'arguments_list': arguments_list,
          'roles_list': roles_list
      }
  }
  return jsonify(res)

# 启动服务 默认开在5000端口

# sanity check route
@app.route('/api/process',methods=['POST'])
def index():
    global result
    data=request.json
    print(data)
    text1=data['text1']
    text2=data['text2']
    p1_start=text1.find("<trigger>")+len("<trigger>")
    p1_end=text1.find("</trigger>")-1
    p2_start=text2.find("<trigger>")+len("<trigger>")
    p2_end=text2.find("</trigger>")-1
    result = data['result']
    trigger_1=text1[p1_start:p1_end+1]
    trigger_2=text2[p2_start:p2_end+1]
    print(trigger_1,trigger_2)
    # enc_text_1 = text1[:p1_start] + '<trigger>' + text1[p1_start:p1_end+1] + '</trigger>' + text1[p1_end+1:]
    # enc_text_2 = text2[:p2_start] + '<trigger>' + text2[p2_start:p2_end + 1] + '</trigger>' + text2[p2_end + 1:]
    enc_text=text1+text2
    model=torch.load('/media/dell/pretrainModel/fnlp/ERE/model.pt')
    print(text1,text2)
    prompt= trigger_1 + '事件与' + trigger_2 + '事件' + '存在' + '[MASK][MASK]' + '关系'
    encoding=tokenizer.encode_plus(enc_text,max_length=512,padding='max_length')
    decoding=tokenizer.encode_plus(prompt,max_length=30,padding='max_length')
    enc_input_ids=torch.tensor([encoding.input_ids],device=torch.device('cuda:0'),dtype=torch.int32)
    enc_mask_ids=torch.tensor([encoding.attention_mask],device=torch.device('cuda:0'),dtype=torch.bool)
    dec_input_ids=torch.tensor([decoding.input_ids],device=torch.device('cuda:0'),dtype=torch.int32)
    dec_mask_ids=torch.tensor([decoding.attention_mask],device=torch.device('cuda:0'),dtype=torch.bool)
    mask_positions = tuple(torch.where(dec_input_ids[0]== tokenizer.mask_token_id)[0])
    mask_positions =torch.tensor([mask_positions])
    targets = torch.tensor([0],device=torch.device('cuda:0'))
    token_ids = torch.tensor([0], device=torch.device('cuda:0'))
    print(mask_positions)
    logits, predictions = model(enc_input_ids, enc_mask_ids, dec_input_ids, dec_mask_ids, mask_positions, targets=targets,token_ids=token_ids, training=False)
    softmax = nn.Softmax(dim=-1)
    probs = softmax(logits)
    predicted_ids = predictions.argmax(dim=-1).tolist()
    predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    print(predicted_text)
    y_predict = torch.argmax(probs, 1).cpu().numpy()
    print(probs)
    # print(logits)
    problist=[{"value":probs[0][0].item(),"name":'因果 CAUSAL'},{"value":probs[0][1].item(),"name":'跟随 FOLLOW'},{"value":probs[0][-1].item(),"name":'无关 NORELATION'}]

    # result={"relation": num_to_label(y_predict[0]),"radio":str(y_predict[0])}
    print(result)
    result = {"relation": problist, "radio": str(y_predict[0])}
    return jsonify(result)
@app.route('/api/result',methods=['GET'])
def get_result():
    global result
    return jsonify(result)

if __name__ == '__main__':
  # 启动服务 默认开在5000端口
  app.run(port="5001",debug=True)