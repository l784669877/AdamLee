from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from db_utils import load_db, add_know_record, ZhipuAIEmbeddings
from langchain_openai import ChatOpenAI
#openai
import os




refine_template = '''1. 你是一名SCL编程代码助手，你负责添加临时变量。

2. 不管是否用得到，都应添加变量i防止变量未定义。因此对于给定的代码，若其未在VAR_TEMP处定义变量i，请为其在VAR_TEMP处添加变量i。不要在VAR_CONSTANT处添加！

3. 仅输出可以直接运行的SCL代码即可(放在```scl和```之间，以便我提取)。不要写其他修饰语。即使与源代码相同保持不变，也要请输出的代码不要省略内容，要求可以直接运行。

4. 你有如下示例：
待审查代码：
FUNCTION_BLOCK "FB_CalculateDayOfYear"
VAR_INPUT
   Year : Int;
   Month : Int;
   Day : Int;
END_VAR

VAR_OUTPUT
   DayOfYear : Int;
   error : Bool;
   status : Word;
END_VAR

VAR_TEMP
   daysPerMonth : ARRAY[1..12] OF Int := [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
   isLeapYear : Bool;
   sumDays : Int;
END_VAR

//省略...

END_FUNCTION_BLOCK

模型审查输出（添加了变量i）：
FUNCTION_BLOCK "FB_CalculateDayOfYear"
VAR_INPUT
   Year : Int;
   Month : Int;
   Day : Int;
END_VAR

VAR_OUTPUT
   DayOfYear : Int;
   error : Bool;
   status : Word;
END_VAR

VAR_TEMP
   daysPerMonth : ARRAY[1..12] OF Int := [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
   isLeapYear : Bool;
   sumDays : Int;
   i : Int;
END_VAR

//省略...

END_FUNCTION_BLOCK'''


def KnowlParser(input):
    '''处理检索结果'''
    result = '\n'
    for doc in input:
        result += f'''{doc.page_content}'''
        result += '\n\n\n'
    return result[:-1]


def to_refine(input, code, llm):

    prompt = ChatPromptTemplate.from_messages(
            [   
                ("system", refine_template),
                ("human", "待审查代码：{code}")
            ])

    pr = StrOutputParser()
    chain = (prompt | llm | pr)

    # # 处理语法
    # embedding = ZhipuAIEmbeddings()
    # knowl_block = load_db(embedding, r'database\knowl\knowl_block')
    # grammar_raw = knowl_block.similarity_search(input, k=5)
    # grammar = KnowlParser(grammar_raw)
    
    
    return chain.invoke({'input':input, 'code': code, })#'grammar': grammar})























if(__name__ == '__main__'):
    zhipuai_api_key="db3b7cdfb3495e9d14f144e43001bc3f.PEmDSwyWyR6ColTL"
    glmair = ChatOpenAI(
         model_name="glm-4-airx",
         openai_api_base="https://open.bigmodel.cn/api/paas/v4",
         openai_api_key= zhipuai_api_key,#generate_token(zhipuai_api_key, 10),
         streaming=False,
         verbose=True
      )
    glm4 = ChatOpenAI(
            model_name="glm-4-0520",
            openai_api_base="https://open.bigmodel.cn/api/paas/v4",
            openai_api_key= zhipuai_api_key,#generate_token(zhipuai_api_key, 10),
            streaming=False,
            verbose=True
        )
    

    test='''FUNCTION_BLOCK "FB_SplitWordIntoBytes"\nVAR_INPUT \n   InputWord : Word;\nEND_VAR\n\nVAR_OUTPUT \n   Byte0 : Byte;\n   Byte1 : Byte;\n   Byte2 : Byte;\n   Byte3 : Byte;\n   average : Real;\nEND_VAR\n\nVAR_TEMP \n   count : Int := 0;\n   sum : DInt := 0;\nEND_VAR\n\nBEGIN\n   // 分解16位二进制数\n   #Byte0 := #InputWord AND 16#000F; // 提取最低的4位\n   #Byte1 := SHR(IN := #InputWord, N := 4) AND 16#000F; // 提取次低的4位\n   #Byte2 := SHR(IN := #InputWord, N := 8) AND 16#000F; // 提取次高的4位\n   #Byte3 := SHR(IN := #InputWord, N := 12) AND 16#000F; // 提取最高的4位\n   \n   // 计算平均值\n   IF #Byte0 <> 0 THEN\n      #sum := #sum + #Byte0;\n      #count := #count + 1;\n   END_IF;\n   IF #Byte1 <> 0 THEN\n      #sum := #sum + #Byte1;\n      #count := #count + 1;\n   END_IF;\n   IF #Byte2 <> 0 THEN\n      #sum := #sum + #Byte2;\n      #count := #count + 1;\n   END_IF;\n   IF #Byte3 <> 0 THEN\n      #sum := #sum + #Byte3;\n      #count := #count + 1;\n   END_IF;\n   \n   // 计算并输出平均值\n   IF #count = 0 THEN\n      #average := 0.0;\n   ELSE\n      #average := #sum / #count;\n   END_IF;\nEND_FUNCTION_BLOCK'''
    print(to_refine(test, test, glmair.bind(temperature=0.6, max_tokens=4000)))
