#openai
import os
from langchain_openai import ChatOpenAI
from utils import SeeWhat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from typing import Union
from db_utils import load_db, add_know_record, ZhipuAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda



breakup_template = '''1. 本项目为智能SCL编程项目，旨在通过生成式AI生成用于编程、监控和维护西门子的自动化系统的一款综合性自动化工程软件平台TIA Portal（Totally Integrated Automation Portal）上的SCL编程代码。SCL编程是TIA Portal中一种广泛应用的高级编程语言，它与Pascal和C语言类似，主要用于需要复杂逻辑控制和数据处理的自动化应用。

2. 什么是任务拆解？在智能SCL编程项目中，任务拆解指的是将用户需求拆解为若干个最小子任务的过程，任务拆解的目标是为了降低完成复杂任务的难度。用户输入是对任务的json形式的描述，你的输出也是子任务的json形式的描述(注意生成的json格式的字符串不要缺少 '｝' 或 '｛' 导致解析失败)。

3. 什么是最小子任务？最小子任务指的是在智能SCL编程项目中，在进行复杂任务拆解时拆解得到的最小任务单元。最小子任务是粗粒度的，要求尽可能的少。

4. 若一个任务拆解成了多个最小子任务，那么请注意这多个最小子任务的任务顺序。

5. 你是一名任务拆解助手，给定你任务的参数的json格式的描述与任务的工作流程，负责智能SCL编程项目中用户需求拆解。当一个用户需求可以拆解为多个最小子任务时，请对其进行最小子任务的拆解；而当一个用户需求本身就是一个最小子任务时，则无需对其进行拆解。

6. 请根据示例的格式输出json，不要输出其他修饰语。你有如下示例：
{few_shot}
'''





class InputParameter(BaseModel):
    name: str = Field(description="The name of the input parameter.")
    type: str = Field(description="The data type of the input parameter.")
    description: str = Field(description="A description of the input parameter.")

class OutputParameter(BaseModel):
    name: str = Field(description="The name of the output parameter.")
    type: str = Field(description="The data type of the output parameter.")
    description: str = Field(description="A description of the output parameter.")

class FunctionBlock(BaseModel):
    title: str = Field(description="The title of the function block.")
    description: str = Field(description="A detailed description of the function block.")
    type: str = Field(description="The type of the function block.")
    name: str = Field(description="The name of the function block.")
    input: List[InputParameter] = Field(description="List of input parameters for the function block.")
    output: List[OutputParameter] = Field(description="List of output parameters for the function block.")


class SubTasks(BaseModel):
    subtask1: FunctionBlock = Field(description="拆分得到的子任务1")
    subtask2: Union[str, FunctionBlock] = Field(description="拆分得到的子任务2")
    subtask3: Union[str, FunctionBlock] = Field(description="拆分得到的子任务3，如果原始任务不需要拆分到三个子任务则为'无'")



def MyParser(input):
    '''处理检索结果'''
    result = '\n'
    for doc in input[:]:
        result += f'''代码要求：{doc.page_content}
代码工作流：{doc.metadata['flow']}

拆分得到的子任务（json格式）：{doc.metadata['answer']}'''
        result += '\n\n\n'
    return result[:-1]



prompt = ChatPromptTemplate.from_messages(
        [   
            ("system", breakup_template),
            ("human", "任务的json形式的描述: {input}")
        ])




def to_breakup(input, llm, k=2):
    pr = StrOutputParser()
    embedding = ZhipuAIEmbeddings()
    breakup_db = load_db(embedding, r'database\breakup\mybreakup_db')
    few_shot_retriever = breakup_db.as_retriever(search_kwargs={'k': k})

    prompt = ChatPromptTemplate.from_messages(
            [   
                ("system", breakup_template),
                ("human", "任务的描述: {input}")
            ])

    few_shot_retrieval = RunnableParallel(
        {'few_shot': few_shot_retriever | RunnableLambda(MyParser),
        'input': RunnablePassthrough()}
    )

    chain = (
        few_shot_retrieval
        | prompt
        | llm | pr)
    
    return chain.invoke(input)









if(__name__ == '__main__'):
    pass