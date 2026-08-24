import sys
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建正确的模型路径
model_base_dir = os.path.join(current_dir, '..', '..', '..', 'pretrain_models')
model_name = os.path.join(model_base_dir, 'multilingual-sentiment-analysis')

# 确保模型路径存在
if not os.path.exists(model_name):
    raise FileNotFoundError(f"Model path does not exist: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}

def predict_sentiment(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # 返回概率和预测的类别
    return probs, torch.argmax(probs, dim=1).cpu().numpy()[0]


def batch_predict(text_path,output_path):
    count = 0
    res_dict_list = []

    df = pd.read_csv(text_path, sep="\t", skiprows=1, names=['content','cluster'])
    content = df.iloc[:, 0]
    x_all_res = list(zip(content, content.apply(lambda x: predict_sentiment(x))))
    for text_res in x_all_res:
        count = count + 1
        text = text_res[0]
        probabilities_prediction = text_res[1]
        probabilities = probabilities_prediction[0]
        prediction = probabilities_prediction[1]
        predicted_sentiment = sentiment_map[prediction]

        res_dict = dict()
        res_dict["text"] = text
        res_dict["probability"] = probabilities[0][prediction].item()
        res_dict["predicted_sentiment"] = predicted_sentiment
        res_dict_list.append(res_dict)

    df_cluster_keywords = pd.DataFrame(res_dict_list, columns=["text", "probability", "predicted_sentiment"])
    df_cluster_keywords.to_excel(output_path, sheet_name="sheet1",
                                 index=False,
                                 header=True)

# data  = "../datasrc/llm_cluster_rd.csv"
# batch_predict(data)
