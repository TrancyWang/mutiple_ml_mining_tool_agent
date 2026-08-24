#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 xuyuan-trial-sentiment-bert-chinese 模型的中文情感分析
支持8种情绪分类：none, disgust, happiness, like, fear, sadness, anger, surprise
"""

import sys
import os

# 设置环境变量以避免多线程冲突导致的段错误（必须在导入torch之前设置）
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from academic_agent.infrastructure.model_paths import default_model_root

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建正确的模型路径
model_name = str(default_model_root() / "xuyuan-trial-sentiment-bert-chinese")

# 确保模型路径存在
if not os.path.exists(model_name):
    raise FileNotFoundError(f"Model path does not exist: {model_name}")

print(f"正在加载模型: {model_name}")

# 自动检测设备：优先MPS (Apple Silicon) > CUDA (NVIDIA GPU) > CPU
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))
print(f"使用设备: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
model = model.to(device)
model.eval()  # 设置为评估模式

# 从配置文件中读取标签映射
id2label = model.config.id2label
label2id = model.config.label2id

# 中文标签映射
chinese_label_map = {
    "none": "无情绪",
    "disgust": "厌恶",
    "happiness": "高兴",
    "like": "喜欢",
    "fear": "恐惧",
    "sadness": "悲伤",
    "anger": "愤怒",
    "surprise": "惊讶"
}


def predict_emotion(texts, top_k=3, batch_size=8):
    """
    预测文本的情绪类别
    
    Args:
        texts: 单个文本字符串或文本列表
        top_k: 返回概率最高的前k个情绪类别
        batch_size: 批处理大小，避免内存溢出
        
    Returns:
        如果输入是单个文本，返回包含预测结果的字典
        如果输入是文本列表，返回预测结果列表
    """
    # 如果是单个字符串，转换为列表
    if isinstance(texts, str):
        texts = [texts]
        single_text = True
    else:
        single_text = False
    
    results = []
    
    # 分批处理
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # 对文本进行编码
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        # 将输入移到正确的设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 预测
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 计算softmax概率
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # 处理批次中的每个文本
        for j, text in enumerate(batch_texts):
            # 获取概率tensor
            prob_tensor = probs[j]
            prob_array = prob_tensor.cpu().numpy()
            
            # 获取top_k个最高概率的索引（使用torch.argsort）
            top_indices = torch.argsort(prob_tensor, descending=True)[:top_k].cpu().numpy()
            
            # 构建结果
            result = {
                "text": text,
                "predictions": []
            }
            
            for idx in top_indices:
                label = id2label[idx]
                probability = prob_array[idx]
                chinese_label = chinese_label_map.get(label, label)
                
                result["predictions"].append({
                    "emotion_en": label,
                    "emotion_zh": chinese_label,
                    "probability": float(probability)
                })
            
            # 主要预测结果
            result["primary_emotion_en"] = result["predictions"][0]["emotion_en"]
            result["primary_emotion_zh"] = result["predictions"][0]["emotion_zh"]
            result["primary_probability"] = result["predictions"][0]["probability"]
            
            results.append(result)
    
    # 如果输入是单个文本，返回单个结果
    if single_text:
        return results[0]
    
    return results


def batch_predict_from_file(input_path, output_path, top_k=3):
    """
    从文件中批量预测情绪
    
    Args:
        input_path: 输入文件路径（CSV或Excel）
        output_path: 输出文件路径
        top_k: 返回概率最高的前k个情绪类别
    """
    print(f"正在读取文件: {input_path}")
    
    # 根据文件扩展名选择读取方式
    if input_path.endswith('.csv'):
        try:
            df = pd.read_csv(input_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, encoding='gbk')
    elif input_path.endswith('.xlsx') or input_path.endswith('.xls'):
        df = pd.read_excel(input_path)
    else:
        raise ValueError("不支持的文件格式，请使用CSV或Excel文件")
    
    # 假设第一列是文本内容
    text_column = df.columns[0]
    texts = df[text_column].dropna().tolist()
    
    print(f"共找到 {len(texts)} 条文本进行分析")
    
    # 批量预测
    results = predict_emotion(texts, top_k=top_k, batch_size=8)
    
    # 构建结果DataFrame
    result_data = []
    for res in results:
        row = {
            "text": res["text"],
            "primary_emotion_en": res["primary_emotion_en"],
            "primary_emotion_zh": res["primary_emotion_zh"],
            "primary_probability": res["primary_probability"]
        }
        
        # 添加top_k预测结果
        for j, pred in enumerate(res["predictions"]):
            row[f"top{j+1}_emotion_en"] = pred["emotion_en"]
            row[f"top{j+1}_emotion_zh"] = pred["emotion_zh"]
            row[f"top{j+1}_probability"] = pred["probability"]
        
        # 添加所有情绪的概率
        for label_en in id2label.values():
            row[f"prob_{label_en}"] = None  # 占位符，后面填充
        
        # 填充所有情绪概率
        pred_dict = {p["emotion_en"]: p["probability"] for p in res["predictions"]}
        all_probs = predict_emotion(res["text"], top_k=8, batch_size=1)["predictions"]
        for p in all_probs:
            row[f"prob_{p['emotion_en']}"] = p["probability"]
        
        result_data.append(row)
    
    df_result = pd.DataFrame(result_data)
    
    # 保存结果
    print(f"正在保存结果到: {output_path}")
    df_result.to_excel(output_path, index=False, engine='openpyxl')
    print("分析完成！")
    
    return df_result


def test_chinese_emotion_analysis():
    """
    使用100条中文测试文本进行情感分析测试
    """
    print("=" * 80)
    print("中文情感分析测试（100条数据 - 8分类）")
    print("=" * 80)
    
    # 创建100条测试文本（涵盖8种情绪类别）
    test_texts = [
        # 高兴 (happiness) - 13条
        "今天天气真好，心情特别愉快！",
        "收到心仪大学的录取通知书，我太开心了！",
        "这个电影太有趣了，笑死我了！",
        "和好朋友一起聚会，真的很快乐。",
        "升职加薪了，感觉人生达到了巅峰！",
        "中了彩票，简直不敢相信！",
        "生日收到很多礼物，好幸福！",
        "考试得了满分，太高兴了！",
        "终于实现了自己的梦想！",
        "今天的阳光真温暖，心情美美哒！",
        "听到好消息，整个人都振奋了！",
        "和家人一起过节，温馨又快乐。",
        "完成了挑战，成就感满满！",
        
        # 喜欢 (like) - 12条
        "我非常喜欢这本书的内容。",
        "这家餐厅的菜很好吃，我经常来。",
        "这个品牌的衣服质量很好，值得信赖。",
        "我喜欢在周末去公园散步。",
        "这款手机的性能很棒，很满意。",
        "老师的讲课方式我很喜欢。",
        "这个地方风景优美，我很中意。",
        "这个音乐很动听，百听不厌。",
        "我对这个项目很感兴趣。",
        "这个设计很有创意，我喜欢。",
        "这家咖啡店的环境很舒服。",
        "这个游戏的玩法很有趣。",
        
        # 愤怒 (anger) - 13条
        "这个服务态度太差了，我非常生气！",
        "等了两个小时还没轮到我，真是气死人了！",
        "这种做法太不公平了，我很愤怒！",
        "被人误解的感觉真的很糟糕，我很恼火。",
        "他居然欺骗我，我气得发抖。",
        "这种不负责任的行为让人愤怒！",
        "凭什么他要这样对我？",
        "我被无理取闹的人惹怒了。",
        "这个决定太荒谬了，我强烈反对！",
        "他的言行让我非常恼火。",
        "这种欺压行为不能容忍！",
        "我被冤枉了，心里很气愤。",
        "这个政策太不合理了，令人愤怒。",
        
        # 悲伤 (sadness) - 13条
        "听到这个消息，我感到非常难过。",
        "失去了重要的朋友，心里空落落的。",
        "考试没考好，感觉很沮丧。",
        "宠物生病了，我很担心也很伤心。",
        "下雨天不能出去玩，有点失落。",
        "离别总是让人伤感。",
        "回忆起往事，心中涌起悲伤。",
        "失去亲人的痛苦难以言表。",
        "孤独的时候总会感到忧伤。",
        "看到感人的电影情节，我哭了。",
        "失败的滋味真不好受。",
        "思念远方的家人，心里很难过。",
        "梦想破灭了，感到很绝望。",
        
        # 恐惧 (fear) - 12条
        "明天就要考试了，我好害怕考不好。",
        "晚上一个人走夜路，感觉很害怕。",
        "听说那里很危险，我不敢去。",
        "面对未知的未来，我感到恐惧。",
        "那只大狗冲我叫，吓死我了。",
        "担心项目不能按时完成，很焦虑。",
        "面试就在明天，我现在很紧张。",
        "等待考试成绩的过程很煎熬。",
        "黑暗让我感到不安和恐惧。",
        "害怕失去现在拥有的一切。",
        "对失败有着深深的恐惧。",
        " heights让我感到头晕目眩，很害怕。",
        
        # 厌恶 (disgust) - 12条
        "这个地方太脏了，让人很不舒服。",
        "他的行为真的很令人反感。",
        "这种食物看起来就很恶心。",
        "看到有人随地吐痰，我觉得很厌恶。",
        "这种虚伪的行为让我感到恶心。",
        "他的言论让人感到不适。",
        "这种不卫生的习惯真让人讨厌。",
        "看到腐败现象，我感到憎恶。",
        "这种低俗的内容令人作呕。",
        "他的做法让人感到鄙夷。",
        "这种浪费行为让人看不下去。",
        "对这种不道德的行为感到厌恶。",
        
        # 惊讶 (surprise) - 13条
        "哇！没想到你会出现在这里！",
        "这个结果完全出乎我的意料！",
        "天哪，他竟然考了满分！",
        "突然收到礼物，太惊喜了！",
        "原来如此，我终于明白了，好惊讶。",
        "看到这个场景我很震惊。",
        "突如其来的消息让我目瞪口呆。",
        "简直不敢相信自己的眼睛。",
        "这个转折太意外了！",
        "完全没想到会是这样的结果。",
        "他的变化太大了，让我很吃惊。",
        "这个消息太让人意外了！",
        "居然发生了这种事，太不可思议了！",
        
        # 无情绪 (none) - 12条
        "今天是星期三，天气多云。",
        "我每天早上七点起床。",
        "这个项目的进展还可以。",
        "中午吃了米饭和蔬菜。",
        "地铁比公交车快一些。",
        "这本书的内容比较一般。",
        "工作还算顺利，没什么特别的。",
        "今天的会议开了一个小时。",
        "路上有点堵车，不过还好。",
        "这个餐厅的菜品中规中矩。",
        "时间过得真快啊。",
        "又过了一天，平淡无奇。",
    ]
    
    print(f"\n开始分析 {len(test_texts)} 条测试文本...\n")
    
    # 批量预测
    results = predict_emotion(test_texts, top_k=3, batch_size=8)
    
    # 打印统计信息
    print("=" * 80)
    print("情绪分布统计")
    print("=" * 80)
    emotion_counts = {}
    for res in results:
        emotion = res["primary_emotion_zh"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(results) * 100
        print(f"{emotion}: {count} 条 ({percentage:.1f}%)")
    
    print("\n" + "=" * 80)
    print("各类别示例展示")
    print("=" * 80)
    
    # 按情绪分类展示示例
    emotion_examples = {}
    for res in results:
        emotion = res["primary_emotion_zh"]
        if emotion not in emotion_examples:
            emotion_examples[emotion] = []
        if len(emotion_examples[emotion]) < 2:  # 每个类别最多展示2个例子
            emotion_examples[emotion].append(res)
    
    for emotion, examples in emotion_examples.items():
        print(f"\n【{emotion}】")
        for ex in examples:
            print(f"  文本: {ex['text']}")
            print(f"  置信度: {ex['primary_probability']:.4f}")
            print(f"  Top 3: ", end="")
            top3 = [f"{p['emotion_zh']}({p['probability']:.2f})" for p in ex['predictions'][:3]]
            print(", ".join(top3))
            print()
    
    # 保存结果到Excel
    output_path = os.path.join(current_dir, 'chinese_emotion_test_results.xlsx')
    result_data = []
    for res in results:
        row = {
            "text": res["text"],
            "primary_emotion_en": res["primary_emotion_en"],
            "primary_emotion_zh": res["primary_emotion_zh"],
            "primary_probability": res["primary_probability"]
        }
        for j, pred in enumerate(res["predictions"], 1):
            row[f"top{j}_emotion_en"] = pred["emotion_en"]
            row[f"top{j}_emotion_zh"] = pred["emotion_zh"]
            row[f"top{j}_probability"] = pred["probability"]
        result_data.append(row)
    
    df_result = pd.DataFrame(result_data)
    df_result.to_excel(output_path, index=False, engine='openpyxl')
    print(f"\n完整结果已保存到: {output_path}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    # 运行100条中文测试
    test_chinese_emotion_analysis()
