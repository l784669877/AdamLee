
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#openai
import os
from langchain_openai import ChatOpenAI
from utils import SeeWhat
from langchain_core.output_parsers import StrOutputParser
from db_utils import load_db, add_know_record, ZhipuAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.schema import Document


def add_analysis_record(db, item):
    """
    添加新的记录，作为后续few-shot的示例。
    :param db: 当前数据库对象
    :param item: 需要添加的知识数据
    :return: None
    """
    ids = db.add_documents([Document(page_content=item[0],
                               metadata={'answer': item[1]})])
    return ids



analysis_template = '''1. 本项目为智能SCL编程项目，旨在通过生成式AI生成用于编程、监控和维护西门子的自动化系统的一款综合性自动化工程软件平台TIA Portal（Totally Integrated Automation Portal）上的SCL编程代码。SCL编程是TIA Portal中一种广泛应用的高级编程语言，它与Pascal和C语言类似，主要用于需要复杂逻辑控制和数据处理的自动化应用。

2. 什么是任务解析？在智能SCL编程项目中，任务解析指的是将用户需求的对应函数需求的工作流程写出来，任务解析的目标是为了使目标函数的工作流程清晰易懂。

3. 你是一名SCL编程代码解析助手，负责解析SCL编程项目的代码。你将解析一段json格式的SCL代码的人物需求，按顺序生成对应的任务流程。用1. 2. 3. (1) (2) (3)这样的标号详细写出代码主要做了什么即这段代码的工作流程。用户输入是对任务的json形式的描述。注意，仅输出工作流程不要写其他修饰语，不同的流程之间用\n\n分隔便于我拆分。

4. 请注意任务流程的顺序和任务的并行性！

5. 你有如下示例：
{few_shot}'''


def MyParser(input):
    '''处理检索结果'''
    result = '\n'
    for doc in input[:]:
        result += '示例输入：\n' + doc.page_content + '\n示例输出：\n' + doc.metadata['answer']
        result += '\n\n\n'
    return result[:-1]




def to_analysis(input, llm, k=2):
    pr = StrOutputParser()
    embedding = ZhipuAIEmbeddings()
    analysis_db = load_db(embedding, r'database\analysis\myanalysis_db')
    analysis_db_few_shot_retriever = analysis_db.as_retriever(search_kwargs={'k': k})

    prompt = ChatPromptTemplate.from_messages(
            [   
                ("system", analysis_template),
                ("human", "任务的json形式的描述: {input}\n\n\n开始！请参考给定的示例的思路。")
            ])
    
    few_shot_retrieval = RunnableParallel(
        {'few_shot': analysis_db_few_shot_retriever | RunnableLambda(MyParser),
        'input': RunnablePassthrough()}
    )


    chain = (
        few_shot_retrieval
        | prompt
        | llm | pr)
    
    return chain.invoke(input)


if(__name__ == '__main__'):
    pass
