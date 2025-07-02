"""
Air Quality Prediction System API

This FastAPI application provides an API for predicting air quality
and retrieving city information.
"""

import os
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import GenAI utilities for image generation
from genai.generate_image import generate_city_poster
from genai.prompt_templates import get_aqi_category, get_health_advice
from ml.model_inference import DeployedModelInference, get_historical_data, DummyModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Air Quality Prediction API",
    description="API for predicting air quality and retrieving city information",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configuration
S3_BUCKET = os.environ.get('S3_BUCKET', 'air-quality-predictor-images')
CITIES_FILE = os.environ.get('CITIES_FILE', './api/data/cities.json')

models = {}


@app.on_event("startup")
async def startup_load_models():
    """启动时加载所有模型"""
    global models
    logger.info("Loading models on startup")
    predictor = DeployedModelInference()
    predictor.load_model()
    models["default"] = predictor


# Load cities data
def load_cities():
    """Load supported cities from JSON file"""
    try:
        if os.path.exists(CITIES_FILE):
            with open(CITIES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Default cities if file doesn't exist
            return {
                "cities": [
                    {"id": "beijing", "name": "北京", "location_id": "123456", "latitude": 39.9042,
                     "longitude": 116.4074},
                    {"id": "shanghai", "name": "上海", "location_id": "234567", "latitude": 31.2304,
                     "longitude": 121.4737},
                    {"id": "guangzhou", "name": "广州", "location_id": "345678", "latitude": 23.1291,
                     "longitude": 113.2644}
                ]
            }
    except Exception as e:
        logger.error(f"Error loading cities data: {e}")
        # Return default cities on error
        return {
            "cities": [
                {"id": "beijing", "name": "北京", "location_id": "123456", "latitude": 39.9042, "longitude": 116.4074},
                {"id": "shanghai", "name": "上海", "location_id": "234567", "latitude": 31.2304, "longitude": 121.4737}
            ]
        }


# Pydantic models for request/response validation
class City(BaseModel):
    id: str
    name: str
    location_id: str
    latitude: float
    longitude: float


class PollutantLevels(BaseModel):
    pm25: float
    pm10: float
    o3: float
    no2: float
    so2: float
    co: float


class PredictionRequest(BaseModel):
    city_id: str
    date: str  # YYYY-MM-DD format


class PredictionResponse(BaseModel):
    aqi: int
    level: str
    pollutants: PollutantLevels
    image_url: str = ""  # 设置默认值为空字符串
    image_data: Optional[str] = None  # 添加可选的image_data字段
    health_advice: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service: str


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "air-quality-predictor"
    }


# Get supported cities
@app.get("/api/cities", response_model=Dict[str, List[City]])
async def get_cities():
    """Get list of supported cities"""
    try:
        cities_data = load_cities()
        return cities_data
    except Exception as e:
        logger.error(f"Error retrieving cities: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cities")


# Predict air quality
@app.post("/api/predict", response_model=PredictionResponse)
async def predict_air_quality(request: PredictionRequest):
    """Predict air quality for a specific city and date"""
    try:
        # Validate input
        try:
            prediction_date = datetime.strptime(request.date, "%Y-%m-%d")
            logger.info(f"Prediction request for date: {prediction_date.isoformat()} and city: {request.city_id}")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

        # 移除日期验证逻辑，允许任何有效日期
        # 只记录一个警告如果日期是过去的
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if prediction_date < today:
            logger.warning(f"Request for past date: {prediction_date.isoformat()}")

        # Get city information
        cities_data = load_cities()
        city = next((c for c in cities_data["cities"] if c["id"] == request.city_id), None)
        if not city:
            raise HTTPException(status_code=404, detail=f"City with ID {request.city_id} not found")

        api_key = os.getenv("OPENAQ_API_KEY")
        if not api_key:
            raise RuntimeError("API key not set in environment variable OPENAQ_API_KEY")
        # Load model and make prediction
        try:

            data_date = prediction_date - timedelta(days=1)
            data_date = data_date.strftime("%Y-%m-%d")

            predictor = models.get("default")

            context_len = predictor.get_context_len()
            try:
                context_len = int(context_len)
            except (ValueError, TypeError):
                context_len = 14

            # 获取历史数据
            historical_data = get_historical_data(
                city_id=request.city_id,
                end_date=data_date,  # 使用到预测日期前一天的数据
                days=context_len,  # 使用context_len天的历史数据
                api_key=api_key
            )

            if historical_data.empty:
                raise HTTPException(status_code=404, detail=f"No historical data found for city {request.city_id}")

            # Make prediction
            aqi_prediction = predictor.predict(historical_data, prediction_date, use_best_model=True).iloc[0]

        except Exception as model_error:
            # 使用默认值并记录错误
            logger.error(f"Error making prediction: {model_error}")
            features = {
                "LATITUDE": city["latitude"],
                "LONGITUDE": city["longitude"],
                "TEMP": 25.0,  # Example temperature
                "DEWP": 15.0,  # Example dew point
                "SLP": 1013.0,  # Example sea level pressure
                "VISIB": 10.0,  # Example visibility
                "WDSP": 5.0,  # Example wind speed
                # Add other required features
            }
            dummy_model = DummyModel("AQI")
            aqi_prediction = dummy_model.predict(features)

        # Round AQI to integer
        aqi_value = int(round(aqi_prediction))

        # Get AQI category and health advice
        aqi_category = get_aqi_category(aqi_value)
        health_advice = get_health_advice(aqi_category)

        # Generate pollutant levels (simplified for demo)
        # In a real implementation, these would come from specific pollutant models
        pollutants = {
            "pm25": round(aqi_value * 0.4, 1),
            "pm10": round(aqi_value * 0.8, 1),
            "o3": round(aqi_value * 0.3, 1),
            "no2": round(aqi_value * 0.2, 1),
            "so2": round(aqi_value * 0.1, 1),
            "co": round(aqi_value * 0.01, 1)
        }

        # Generate image using GenAI service
        try:
            # 使用请求中的日期作为主题，而不是固定的"Air Quality Visualization"
            theme = request.date  # 使用YYYY-MM-DD格式的日期作为主题

            image_result = generate_city_poster(
                city_name=city["name"],
                theme_of_day=theme,
                aqi_value=aqi_value,
                s3_bucket=S3_BUCKET
            )
            # 确保image_url始终是字符串，即使s3_url是None
            image_url = image_result.get("s3_url") or ""
            # 获取base64编码的图像数据
            image_data = image_result.get("image_data")
        except Exception as image_error:
            logger.error(f"Error generating image: {image_error}")
            image_url = ""  # 如果图像生成失败，使用空URL
            image_data = None

        # Return prediction response
        return {
            "aqi": aqi_value,
            "level": aqi_category,
            "pollutants": pollutants,
            "image_url": image_url,
            "image_data": image_data,
            "health_advice": health_advice
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict_air_quality: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Run the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
