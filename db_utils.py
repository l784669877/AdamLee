import os
from tqdm import tqdm
from typing import Annotated, Literal, Optional, List

from langchain.schema import Document
from langchain_chroma import Chroma
import numpy as np
from typing import Dict, List, Any
from langchain_core.embeddings import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator






class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """`Zhipuai Embeddings` embedding models."""
    
    client: Any
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """
        实例化ZhipuAI为values["client"]
        Args:
            values (Dict): 包含配置信息的字典，必须包含 client 的字段.
        Returns:
            values (Dict): 包含配置信息的字典。如果环境中有zhipuai库，则将返回实例化的ZhipuAI类；否则将报错 'ModuleNotFoundError: No module named 'zhipuai''.
        """
        from zhipuai import ZhipuAI
        values["client"] = ZhipuAI(api_key='5c99336a031d732a811759e26c9aa638.g6g3LEJJJEQhZqAN')
        return values
 
 
    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.
        Args:
            texts (str): 要生成 embedding 的文本.
        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """
        embeddings = self.client.embeddings.create(
            model="embedding-2",
            input=text
        )
        return embeddings.data[0].embedding
 
 
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.
        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        return [self.embed_query(text) for text in texts]
 
 
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError("Please use `embed_documents`. Official does not support asynchronous requests")
 
    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError("Please use `aembed_query`. Official does not support asynchronous requests")
    



def add_know_record(db, item):
    """
    添加新的记录，作为后续few-shot的示例。
    :param db: 当前数据库对象
    :param item: 需要添加的知识数据
    :return: None
    """
    ids = db.add_documents([Document(page_content=item[0],
                               metadata={'answer': item[1]})])
    return ids



def load_db(embedding, persist_directory):
    if not os.path.exists(persist_directory):
        document = Document(page_content='1')
        db = Chroma.from_documents([document], embedding, persist_directory=persist_directory)
        db.delete(db.get()['ids'])
    else: # 从已有数据中加载
        db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    return db
