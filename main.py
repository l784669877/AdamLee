from utils import list_json_files
import json
#openai
import os
from langchain_openai import ChatOpenAI
import jwt
import time
from generate import to_generate, add_gen_record
from analysis import to_analysis, add_analysis_record
from breakup import to_breakup
from refine import to_refine
from refine2 import to_refine2,add_refine_record
from db_utils import load_db, add_know_record, ZhipuAIEmbeddings
from generate import to_generate_sub
import re
import time










zhipuai_api_key="db3b7cdfb3495e9d14f144e43001bc3f.PEmDSwyWyR6ColTL"
def generate_token(apikey: str, exp_seconds: int):
        id, secret = apikey.split(".")
        payload = {
            "api_key": id,
            "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
            "timestamp": int(round(time.time() * 1000)),
        }

        return jwt.encode(
            payload,
            secret,
            algorithm="HS256",
            headers={"alg": "HS256", "sign_type": "SIGN"},
        )

glm4 = ChatOpenAI(
        model_name="glm-4-0520",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_key= zhipuai_api_key,#generate_token(zhipuai_api_key, 10),
        streaming=False,
        verbose=True
    )
glmlong = ChatOpenAI(
        model_name="glm-4-long",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_key= zhipuai_api_key,#generate_token(zhipuai_api_key, 10),
        streaming=False,
        verbose=True
    )
glmair = ChatOpenAI(
        model_name="glm-4-airx",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_key= zhipuai_api_key,#generate_token(zhipuai_api_key, 10),
        streaming=False,
        verbose=True
    )
codegeex = ChatOpenAI(
    model_name="codegeex-4",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=zhipuai_api_key,
    streaming=False,
    verbose=True
)



def do_subtask(subtasks, llm1, llm2):
    
    my_few_shot = ''

    for i, subtask in enumerate(subtasks):
        if(subtask == '无'):
             continue
        subtask_analysis = to_analysis(str(subtask), llm1.bind(temperature=0.5, max_tokens=4000), k=5)
        subtask_input = f'''代码要求：{str(subtask)}
代码工作流：{subtask_analysis}\n'''
        
        _subtask_code = to_generate_sub(str(subtask), subtask_analysis, llm2.bind(temperature=0.6, max_tokens=4000), k=2)

        
        pattern = re.compile(rf'{re.escape("```scl")}(.*?){re.escape("```")}', re.DOTALL)
        match = pattern.search(_subtask_code)
        if(match):
            subtask_code = match.group(1).strip()
        else:
            subtask_code = _subtask_code

        
    
        my_few_shot += f'''示例输入：
代码要求：{subtask}
代码工作流：{subtask_analysis}

示例输出：{subtask_code}\n\n'''
        
    return my_few_shot


embedding = ZhipuAIEmbeddings()
refine_db = load_db(embedding, r'database\refine\refine_db')
generate_db = load_db(embedding, r'database\generate\generate_db')
analysis_db = load_db(embedding, r'database\analysis\myanalysis_db')

os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_ENDPOINT"] = 'https://api.smith.langchain.com'
os.environ["LANGCHAIN_API_KEY"] = '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'

if __name__ == '__main__':


    
    with open('code_blocks.jsonl', 'w', encoding='utf-8-sig') as lfile:

        myques = list_json_files(r'sample program')
        for file in myques[:]:
            file_name_with_extension = os.path.basename(file)
            file_name, file_extension = os.path.splitext(file_name_with_extension)

            start_time = time.time()

            print(file)
            with open(file, 'r', encoding='utf-8-sig') as jfile:
                myjson = json.load(jfile)
            json_str = json.dumps(myjson, ensure_ascii=False)
        
            # analysis_result = to_analysis(json_str, glmair.bind(temperature=0.5, max_tokens=4000), k=3)
            analysis_result = '不分析'

            _code, f, s = to_generate(json_str, analysis_result,
                                      codegeex.bind(temperature=0.5,
                                                    max_tokens=4000,
                                                    top_p = 0.7), k=5)
            # print(_code, end='\n\n\n')
            pattern = re.compile(rf'{re.escape("```scl")}(.*?){re.escape("```")}', re.DOTALL)
            matches = pattern.findall(_code)
            if(matches):
                code = matches[-1].strip()
            else:
                code = _code

            _second = to_refine2(code, json_str, codegeex.bind(temperature=0.7, max_tokens=4000), f, s)
            pattern = re.compile(rf'{re.escape("```scl")}(.*?){re.escape("```")}', re.DOTALL)
            matches = pattern.findall(_second)
            if matches:
                code = matches[-1].strip()
            else:
                code = _second
            #code = re.sub(r'RETURN', f'#{myjson["name"]} :=', code)

            
            # _last = to_refine('input', code, glmair.bind(temperature=0.3, max_tokens=4000))
            # pattern = re.compile(rf'{re.escape("```scl")}(.*?){re.escape("```")}', re.DOTALL)
            # matches = pattern.findall(_last)
            # if(matches):
            #     last = matches[-1].strip()
            # else:
            #     last = _last
            last = code
            last = re.sub(r'VAR_CONSTANT', 'VAR CONSTANT', last)
            last = re.sub(r'DIV', r'/', last)
            last = re.sub(r'date : DTL;', r'"date" : DTL;', last)
            last = re.sub(r'ELSEIF', r'ELSIF', last)
            last = re.sub(r'RETURN\s+[^;]+;', 'RETURN;', last)

            #last = re.sub(r'RETURN', f'#{myjson["name"]} :=', last)

             
            


            end_time = time.time()
            # 将字符串 'last' 转换为 JSON 格式并写入文件
            my = {"name":file_name, 'code':last}
            lfile.write(json.dumps(my, ensure_ascii=False) + '\n') 



    #         添加到知识库
    #         add_analysis_record(analysis_db, (json_str, analysis_result))
    #         add_gen_record(generate_db, (json_str, last), analysis_result)
    #         add_refine_record(f'''代码要求：
    # {json_str}

    # 代码：
    # {code}

    # 以上代码有算法思路问题。请你检查一下，写出详细检查的过程。请输出完整代码不要用//省略代码的内容''', _last)
            
            # 保存
            match1 = file[:-5]
            txt_file = match1 + '.txt'
            with open(txt_file, 'w', encoding='utf-8-sig') as tfile:
                            
                tfile.write(last)
                tfile.write('\n\n\n-\n\n')
                tfile.write(analysis_result)
                tfile.write('\n\n\n-\n\n')
                tfile.write(str(end_time-start_time))






