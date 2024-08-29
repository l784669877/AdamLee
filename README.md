# 项目名称

## 环境配置

在本项目中，我们使用了以下软件依赖项，并将其详细信息记录在 `requirements.txt` 文件中：

- Python 3.11.8：这是主要编程语言版本，选择此版本是因为其最新的功能和性能优化。

请确保您已按照 `requirements.txt` 文件中的指示安装所有必要的依赖项。

## 代码启动指南

### 步骤 1：设置环境

1. 创建并激活虚拟环境

2. 安装依赖项：
    ```bash
    pip install -r requirements.txt
    ```

### 步骤 2：配置路径

在 `main.py` 文件中，请将 `myques = list_json_files(r'sample program\初赛题目')` 中的路径替换为存放初赛题目的 JSON 文件的绝对路径。例如：
```python
myques = list_json_files(r'/path/to/your/json/files')
```

### 步骤 3：运行代码

0. 更盖好路径后运行main.py文件即可得到对应的答案（txt格式）

1. 运行代码得到符合初赛提交要求的 `jsonl` 。jsonl将存放于main.py相同文件夹下，而不是在初赛题目的 JSON 文件下。

## SCL代码生成方式

### 步骤 1：解析输入的 JSON 文件

在 `analysis.py` 文件中，根据输入的 JSON 文件进行主要任务流程的解析。例如，对于 `FB_BottleProcessing.json`，输出如下：

1. **初始化**：在程序开始时，将所有输出信号初始化为`FALSE`。
2. **瓶子检测**：当`bottleSensor`检测到瓶子时，将`Pump_Motor`设置为`TRUE`以启动清洗泵。
3. **清洗完成确认**：当操作员按下`cleaningConfirmButton`时，将`Pump_Motor`设置为`FALSE`以停止清洗泵，并将`Filling_Valve`设置为`TRUE`以启动灌装阀。
4. **灌装完成确认**：当操作员按下`fillingConfirmButton`时，将`Filling_Valve`设置为`FALSE`以停止灌装阀，并将`Capping_Machine`设置为`TRUE`以启动封盖机。
5. **封盖完成确认**：当操作员按下`cappingConfirmButton`时，将`Capping_Machine`设置为`FALSE`以停止封盖机，并将`Packing_Machine`设置为`TRUE`以启动包装机。
6. **包装完成确认**：当操作员按下`packingConfirmButton`时，将`Packing_Machine`设置为`FALSE`以停止包装机，并将`Completion_Light`设置为`TRUE`以点亮包装完成指示灯。
7. **复位指示灯**：当操作员取走包装好的瓶子并按下`finishedButton`时，将`Completion_Light`设置为`FALSE`以关闭指示灯，并准备开始下一个瓶子的生产循环。

### 步骤 2：生成 SCL 代码

1. 根据解析结果，直接生成对应的 SCL 代码。初始示例 `few-shot` 由15个示例gt提供，相似度检索和语法知识库、函数知识库根据给定的 PDF 文件解析获得。

### 步骤 3：优化 SCL 代码

1. 生成的 SCL 代码可能存在问题，因此需要进行处理。优化分为两个过程，分别由 `refine.py` 和 `refine2.py` 文件完成。