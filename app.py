from flask import Flask, request, jsonify
import logging
from logging.handlers import RotatingFileHandler
import json
import re
from langchain_openai import ChatOpenAI
from generate import to_generate, add_gen_record
from refine2 import to_refine2, add_refine_record
from db_utils import load_db, add_know_record, ZhipuAIEmbeddings


zhipuai_api_key = "db3b7cdfb3495e9d14f144e43001bc3f.PEmDSwyWyR6ColTL"
glm4 = ChatOpenAI(
    model_name="glm-4-0520",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=zhipuai_api_key,
    streaming=False,
    verbose=True
)
glmlong = ChatOpenAI(
    model_name="glm-4-long",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=zhipuai_api_key,
    streaming=False,
    verbose=True
)
glmair = ChatOpenAI(
    model_name="glm-4-airx",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=zhipuai_api_key,
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

embedding = ZhipuAIEmbeddings()
refine_db = load_db(embedding, r'database/refine/refine_db')
generate_db = load_db(embedding, r'database/generate/generate_db')
analysis_db = load_db(embedding, r'database/analysis/myanalysis_db')

app = Flask(__name__)

# 配置日志记录器
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

# 设置 Flask 应用的日志级别
app.logger.setLevel(logging.ERROR)

@app.route('/', methods=['POST'])
def generate_code():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400
        input_str = json.dumps(data, indent=4)
        input_str = input_str.replace("\\\\", "\\")
        data = json.loads(input_str)

        
        # try:
        #     data = json.loads(json.dumps(data), ensure_ascii=False)
        # except Exception as e:
        #     app.logger.error(f"Error occurred: {str(e)}")
        #     return jsonify({"error": "Internal Server Error"}), 500

        app.logger.info(f"Received data: {data}")

        try:
            json_str = json.dumps(data, ensure_ascii=False, indent=4)
        except json.JSONDecodeError as e:
            app.logger.error(f"JSON decode error: {str(e)}")
            return jsonify({"error": "Invalid JSON format"}), 400

        # 模拟代码生成逻辑
        analysis_result = '不分析'

        app.logger.info("Generating code...")
        _code, f, s = to_generate(json_str, analysis_result,
                                      codegeex.bind(temperature=0.5,
                                                    max_tokens=4000,
                                                    top_p = 0.7), k=4)
        app.logger.info(f"Generated code: {_code}")
        pattern = re.compile(rf'{re.escape("```scl")}(.*?){re.escape("```")}', re.DOTALL)
        match = pattern.search(_code)
        if match:
            code = match.group(1).strip()
        else:
            code = _code


        app.logger.info("Refining code...")
        _second = to_refine2(code, json_str, codegeex.bind(temperature=0.7, max_tokens=4000), f, s)
        app.logger.info(f"Refined code: {_second}")
        pattern = re.compile(rf'{re.escape("```scl")}(.*?){re.escape("```")}', re.DOTALL)
        matches = pattern.findall(_second)
        if matches:
            code = matches[-1].strip()
        else:
            code = _second

        last = code
        last = re.sub(r'VAR_CONSTANT', 'VAR CONSTANT', last)
        last = re.sub(r'DIV', r'/', last)
        last = re.sub(r'date : DTL;', r'"date" : DTL;', last)
        last = re.sub(r'ELSEIF', r'ELSIF', last)
        last = re.sub(r'RETURN\s+[^;]+;', 'RETURN;', last)

        response = {
            "name": data.get("name"),
            "code": last
        }
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error occurred: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)