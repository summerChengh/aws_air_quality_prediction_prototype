"""
FastAPI service for air quality prediction model inference
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from model_inference import DeployedModelInference, get_historical_data

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(
    title="Air Quality Prediction API",
    description="API for predicting air quality metrics using trained models",
    version="1.0.0"
)

# 预加载模型
models = {}


@app.on_event("startup")
async def startup_load_models():
    """启动时加载所有模型"""
    global models
    logger.info("Loading models on startup")
    predictor = DeployedModelInference()
    predictor.load_model()
    models["default"] = predictor


# 定义API数据模型
class PredictionRequest(BaseModel):
    city_id: str = Field(..., description="City identifier")
    date: Optional[str] = Field(None, description="Prediction date (YYYY-MM-DD), defaults to tomorrow")
    features: Optional[Dict[str, float]] = Field(None, description="Additional feature values")
    pollutants: Optional[List[str]] = Field(None, description="Specific pollutants to predict")


class PredictionResponse(BaseModel):
    city_id: str
    prediction_date: str
    predictions: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    available_pollutants: List[str]
    timestamp: str


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """预测空气质量指标"""
    try:
        # 确定预测日期
        if request.date:
            prediction_date = datetime.strptime(request.date, "%Y-%m-%d")
        else:
            # 默认预测明天
            prediction_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        predictor = models.get("default")

        context_len = predictor.get_context_len()
        if isinstance(context_len, str) and context_len.isdigit():
            context_len = int(context_len)
        else:
            context_len = 14
        api_key = os.getenv("OPENAQ_API_KEY")
        if not api_key:
            raise RuntimeError("API key not set in environment variable OPENAQ_API_KEY")
        # 获取历史数据
        historical_data = get_historical_data(
            city_id=request.city_id,
            end_date=prediction_date - timedelta(days=1),  # 使用到预测日期前一天的数据
            days=context_len,  # 使用context_len天的历史数据
            api_key=api_key
        )

        if historical_data.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for city {request.city_id}")

        # 存储预测结果
        predictions = {}

        predict_AQI = predictor.predict(historical_data, prediction_date, use_best_model=True).iloc[0]

        predictions["AQI"] = predict_AQI

        # 构建响应
        response = {
            "city_id": request.city_id,
            "prediction_date": prediction_date.strftime("%Y-%m-%d"),
            "predictions": predictions,
        }

        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models")
async def list_models():
    """列出所有可用模型"""
    return {
        "available_models": list(models.keys())
    }


if __name__ == "__main__":
    # 运行服务
    uvicorn.run(app, host="0.0.0.0", port=8000)
