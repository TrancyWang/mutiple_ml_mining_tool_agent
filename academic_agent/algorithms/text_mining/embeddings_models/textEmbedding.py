#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: TrancyWang
LastEditors: TrancyWang
Date: 2020-08-21 17:25:40
LastEditTime: 2020-08-27 22:11:38
Desciption: SentenceEmbedding
'''
from sentence_transformers import SentenceTransformer
import torch
import logging
import  os
from academic_agent.infrastructure.model_paths import default_model_root

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


class TextEmbedding(object):
    """
      基于sentencebert+simbert finetune 进行编码
    """
    def __init__(self):
        ###加载Financial finetune后的sentencebert模型
        self.embedding_model = str(default_model_root() / "bge-cn")
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        # 检查模型路径是否存在
        if not os.path.exists(self.embedding_model):
            raise FileNotFoundError(f"模型路径不存在: {self.embedding_model}")

        try:
            self.bert = SentenceTransformer(
                self.embedding_model,
                device=self.device,
                use_safetensors=True  # 优先使用safetensors格式
            )
        except:
            # 如果不支持，则使用默认方式
            self.bert = SentenceTransformer(
                self.embedding_model,
                device=self.device
            )
            logger.warning("device is ...  " + str(self.device))

    def get_query_vector (self, query):
        query_vector = None
        try:
         query_vector = self.bert.encode(query,batch_size=64)
        except Exception as e:
            logger.error(f"Error processing query: {query}")
            logger.error(f"Error details: {e}")
        return query_vector
