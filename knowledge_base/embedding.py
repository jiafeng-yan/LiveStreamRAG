from typing import List
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


class EmbeddingModel:
    """文本嵌入模型封装类"""

    def __init__(self, model_name="bert-base-uncased"):
        """
        初始化嵌入模型

        Args:
            model_name: 使用的模型名称
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def _mean_pooling(self, model_output, attention_mask):
        """平均池化操作"""
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将多个文本转换为嵌入向量"""
        # 分批处理以节省内存
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            # 计算token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # 池化
            sentence_embeddings = self._mean_pooling(
                model_output, encoded_input["attention_mask"]
            )

            # 归一化
            sentence_embeddings = torch.nn.functional.normalize(
                sentence_embeddings, p=2, dim=1
            )

            all_embeddings.extend(sentence_embeddings.cpu().numpy().tolist())

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """将单个查询文本转换为嵌入向量"""
        return self.embed_documents([text])[0]
