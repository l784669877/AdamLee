import json
import os
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID
import json
import os





class SeeWhat(BaseCallbackHandler):
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        print(f"the inputs of the chain is {inputs}\n\n")
    
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"the messages of the llm is {messages}\n\n")

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"the prompts of the llm is {prompts}\n\n")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"the input_str of the tool is {input_str}\n\n")












def insert_item_to_json(file_path, new_item):
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 读取现有的 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    else:
        # 如果文件不存在，创建一个新的空数组
        data = {}

    # 插入新的项目
    data[new_item[0]] = new_item[1]

    # 将更新后的数据写回到 JSON 文件
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)



def list_json_files(directory = 'sample program'):
    json_files = []
    for file in os.listdir(directory):
        if file.endswith('.json'):
            json_files.append(os.path.join(directory, file))
    return json_files

def list_scl_files(directory = 'sample program'):
    scl_files = []
    for file in os.listdir(directory):
        if file.endswith('.scl'):
            scl_files.append(os.path.join(directory, file))
    return scl_files



def extract_between_braces(input_string):
    start_index = input_string.find('{')
    end_index = input_string.rfind('}')
    
    if start_index == -1 or end_index == -1 or start_index > end_index:
        return ""  # 如果没有找到 { 或 }，或者 { 在 } 之后，返回空字符串
    
    return input_string[start_index: end_index+1]