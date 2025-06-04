import logging
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Логування
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('application.log', mode='a')
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Ініціалізація FastAPI
app = FastAPI(title="Network Traffic Classifier API", version="1.0")

# Завантаження моделей та скейлера
try:
    api_scaler = joblib.load('models/scaler.pkl')
    api_feature_columns = joblib.load('models/feature_columns.pkl')
    api_model_type = joblib.load('models/model_type.pkl')
    if api_model_type == 'Ensemble':
        api_xgb_model = joblib.load('models/final_xgb_model.pkl')
        api_nn_model = tf.keras.models.load_model('models/neural_network_model.keras')
        logger.info("API: Завантажено scaler, feature_columns, XGBoost і Neural Network моделі для ансамблю.")
    else:
        api_xgb_model, api_nn_model = None, None
        logger.error("API: Очікувався тип моделі 'Ensemble'. Моделі не завантажені.")
except FileNotFoundError:
    logger.warning("API: Один або декілька файлів моделі/скейлера (.pkl, .keras) не знайдено.")
    api_scaler, api_feature_columns, api_model_type, api_xgb_model, api_nn_model = None, None, None, None, None
except Exception as e:
    logger.error(f"API: Помилка завантаження моделі/скейлера: {e}")
    api_scaler, api_feature_columns, api_model_type, api_xgb_model, api_nn_model = None, None, None, None, None

class InputData(BaseModel):
    features: dict

@app.post("/predict")
async def predict(data: InputData):
    if not all([api_scaler, api_feature_columns, api_xgb_model, api_nn_model, api_model_type == 'Ensemble']):
        raise HTTPException(status_code=503,
                            detail="Моделі або необхідні компоненти не завантажені. Будь ласка, спочатку натренуйте моделі.")

    try:
        # Перевірка наявності всіх необхідних фіч
        input_features = set(data.features.keys())
        required_features = set(api_feature_columns)

        # Знаходимо відсутні фічі
        missing_features = required_features - input_features
        if missing_features:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Відсутні обов'язкові фічі",
                    "missing_features": list(missing_features),
                    "required_features": api_feature_columns,
                    "provided_features": list(input_features)
                }
            )

        # Перевірка на зайві фічі
        extra_features = input_features - required_features
        if extra_features:
            logger.warning(f"Отримано зайві фічі, які будуть проігноровані: {list(extra_features)}")

        # Створюємо DataFrame тільки з необхідними фічами у правильному порядку
        input_df = pd.DataFrame([{col: data.features[col] for col in api_feature_columns}])

        # Масштабування фіч
        features_scaled = api_scaler.transform(input_df)

        # Передбачення від XGBoost
        xgb_proba = api_xgb_model.predict_proba(features_scaled)[0, 1]
        # Передбачення від нейронної мережі
        nn_proba = api_nn_model.predict(features_scaled, verbose=0)[0, 0]
        # М'яке голосування: середнє ймовірностей
        ensemble_proba = (xgb_proba + nn_proba) / 2
        ensemble_label = 1 if ensemble_proba > 0.5 else 0

        return {
            "prediction_label": int(ensemble_label),
            "prediction_probability_attack": float(ensemble_proba),
            "xgb_probability": float(xgb_proba),
            "nn_probability": float(nn_proba),
            "status": "Attack" if ensemble_label == 1 else "Normal"
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Помилка у вхідних даних: {str(e)}")
    except Exception as e:
        logger.error(f"Помилка під час передбачення: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Внутрішня помилка сервера: {str(e)}")

@app.get("/required-features")
async def get_required_features():
    """Повертає список всіх необхідних фіч для передбачення"""
    if not api_feature_columns:
        raise HTTPException(status_code=503, detail="Модель не завантажена")

    return {
        "required_features": api_feature_columns,
        "total_features_count": len(api_feature_columns),
        "description": "Всі ці фічі є обов'язковими для точного передбачення"
    }