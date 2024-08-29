import os  
import json  
import logging  
from concurrent.futures import ThreadPoolExecutor  
from datetime import datetime  
import requests  

# 接口地址和端口范围  
base_url = 'http://localhost:'  
port = 8000

# 日志  
logging.basicConfig(filename=f'{port}.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
  
# 存储路径  
submissions_dir = 'submissions'  
  

def send_request(json_file_path, port):  
    """  
    发送POST请求并处理响应。  
    """  
    url = f"{base_url}{port}"  
    try:  
        with open(json_file_path, 'r', encoding='utf-8-sig') as file:  
            data = json.load(file)
        headers = {'Content-Type': 'application/json'}  
        response = requests.post(url, json=data, headers=headers, timeout=90)  
        logging.error(f"processing {json_file_path}")     
        # 处理响应  
        if response.status_code == 200:  
            content = response.json()  
            if content in [None, [], {}]:  
                content = {"error": "Empty result"}  
                logging.warning(f"Received empty result from {url} for file {json_file_path}") 
            # 存储数据  
            today = datetime.now().strftime('%Y-%m-%d')  
            file_name = f"{os.path.splitext(os.path.basename(json_file_path))[0]}.json"  
            storage_path = os.path.join(submissions_dir, f'{today}_{port}')  
            os.makedirs(storage_path, exist_ok=True)  
            result_path = os.path.join(storage_path, file_name)  
            with open(result_path, 'w', encoding='utf-8-sig') as f:  
                json.dump(content, f)  
        else:  
            logging.error(f"Request to {url} failed with status code {response.status_code} for file {json_file_path}")  
    except requests.exceptions.RequestException as e:  
        logging.error(f"Network request error for {url} and file {json_file_path}: {str(e)}")  
    except Exception as e:  
        logging.error(f"Unexpected error processing {json_file_path}: {str(e)}")   
  
def main():  
    # 题目
    source_dir = r'E:\STUDY\0.2langchain\LangChain实战\AImail\competition(noanalys)\sample program' 

    json_files = [os.path.join(source_dir, file) for file in os.listdir(source_dir) if file.endswith('.json')]  
    for json_file in json_files: 
        send_request(json_file, port)

  
if __name__ == "__main__":  
    main()
