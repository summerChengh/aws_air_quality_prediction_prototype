# AWS Air Quality Predictor

基于AWS云服务的全球城市空气质量预测系统，集成OpenAQ和NOAA数据，提供高精度空气质量预测和可视化服务。

## 项目概述

本项目利用AWS云服务构建了一个完整的空气质量预测系统，通过整合OpenAQ空气质量数据和NOAA气象数据，结合先进的机器学习技术，为用户提供准确的空气质量预测和个性化的可视化内容。

### 核心功能

- **多源数据整合**: 自动采集和处理来自OpenAQ和NOAA的大规模数据
- **高精度预测**: 利用AutoML技术训练和优化空气质量预测模型
- **时空分析**: 提供基于地理位置和时间维度的空气质量分析
- **个性化内容**: 使用生成式AI创建与城市和空气质量相关的图片
- **API服务**: 提供RESTful API接口供第三方应用集成

## 系统架构

系统采用AWS无服务器架构：


详细架构设计请参阅 [架构文档](docs/DESIGN.md)。

## 技术栈

- **AWS云服务**:
  - Amazon S3: 数据湖存储
  - AWS Glue: ETL数据处理
  - Amazon SageMaker: 机器学习模型开发和部署
  - AWS Lambda: 无服务器计算
  - Amazon API Gateway: API管理
  - Amazon Bedrock: 生成式AI服务
  
- **开发技术**:
  - Python: 主要开发语言
  - PySpark: 大规模数据处理
  - AutoGluon: 自动机器学习框架
  - Vue 3: 前端开发
  - FastAPI: API服务
  - Jupyter/StudioLab: 数据分析与可视化

## 快速开始

### 前提条件

- AWS账户
- Python 3.8+
- AWS CLI已配置

### 安装依赖

1. 克隆仓库:
   ```bash
   git clone https://github.com/yourusername/aws-air-quality-predictor.git
   cd aws-air-quality-predictor
   ```

2. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```
   主要依赖包括：
   - fastapi, uvicorn, pydantic, requests, pandas, numpy, boto3, scikit-learn, autogluon, matplotlib, python-multipart, pillow, aiofiles, pytest, httpx, pytest-asyncio, python-dotenv, jinja2, seaborn, ipywidgets, joblib, PyYAML, configparser, python-dateutil, tqdm

3. 配置AWS凭证:
   ```bash
   aws configure
   ```

4. （可选）Jupyter/StudioLab环境支持：
   - 直接在AWS StudioLab或本地Jupyter Notebook中运行数据分析/可视化代码。

### 环境变量与API密钥管理

- 推荐将敏感信息（如OpenAQ API KEY）存放在`.env`文件中：
  ```
  OPENAQ_API_KEY=your_api_key_here
  ```
- 在Python代码中用`python-dotenv`自动加载：
  ```python
  import os
  from dotenv import load_dotenv
  load_dotenv()
  api_key = os.getenv("OPENAQ_API_KEY")
  ```
- 生产环境建议用云Secret Manager或CI/CD注入环境变量。
- `.env`文件应加入`.gitignore`，避免泄露。

### 数据上传到AWS StudioLab/Jupyter

- 通过Jupyter Lab网页界面左上角"Upload"按钮上传本地数据文件。
- 或在Notebook中用如下代码上传小文件：
  ```python
  from IPython.display import display
  import ipywidgets as widgets
  uploader = widgets.FileUpload(accept='', multiple=False)
  display(uploader)
  # 上传后保存
  import io
  for filename, fileinfo in uploader.value.items():
      with open(filename, 'wb') as f:
          f.write(fileinfo['content'])
  ```

### 典型数据探索分析（EDA）流程示例

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('aqi_data.csv')
print(df.info())
print(df.describe())

# 缺失比例
missing_ratio = df.isnull().mean().sort_values(ascending=False)
print(missing_ratio)

# 缺失可视化
sns.heatmap(df.isnull(), cbar=False)
plt.show()

# 数值特征分布
num_cols = df.select_dtypes(include='number').columns
df[num_cols].hist(figsize=(15, 10), bins=30)
plt.show()

# 相关性热力图
sns.heatmap(df.corr(), annot=True, fmt='.2f')
plt.show()
```

### API服务安全获取API KEY

- FastAPI服务中推荐如下方式安全获取API KEY：
  ```python
  import os
  from dotenv import load_dotenv
  load_dotenv()
  api_key = os.getenv("OPENAQ_API_KEY")
  if not api_key:
      raise RuntimeError("API key not set in environment variable OPENAQ_API_KEY")
  ```
- 不要将API KEY硬编码在代码中。
- 生产环境可用AWS Secrets Manager等云服务管理密钥。

## 项目结构

```
aws-air-quality-predictor/
├── api/                      # API服务
│   ├── main.py               # FastAPI主应用
│   ├── run.py                # API启动脚本
│   └── data/                 # API数据文件
├── data/                     # 数据目录
├── data_ingestion/           # 数据摄取模块
│   ├── fetch_openaq_data.py  # OpenAQ数据获取
│   ├── fetch_noaa_data.py    # NOAA数据获取
│   ├── data_process.py       # 数据处理
│   ├── cal_aqi.py            # AQI计算
│   ├── openaq_utils.py       # OpenAQ工具函数
│   ├── openaq_feats.py       # OpenAQ特征工程
│   └── utils.py              # 通用工具函数
├── frontend/                 # 前端Vue应用
│   ├── src/                  # 源代码
│   ├── public/               # 静态资源
│   └── dist/                 # 构建输出
├── genai/                    # 生成式AI模块
├── ml/                       # 机器学习模块
│   ├── model_inference.py    # 模型推理
│   ├── app.py                # ML服务应用
│   └── automl_train_evaluate.py # 自动机器学习训练评估
├── models/                   # 模型存储
│   ├── deploy/               # 部署模型
│   └── automl/               # AutoML模型
├── notebooks/                # Jupyter笔记本
│   └── exploration.ipynb     # 数据探索分析
├── .env                      # 环境变量（不提交到Git）
├── requirements.txt          # Python依赖
└── README.md                 # 项目说明
```


