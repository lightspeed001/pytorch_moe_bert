import os
import logging
from typing import Optional
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentor

# Configure Logging
logging.basicConfig(
  level=os.getenv("LOG_LEVEL", "INFO"),
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#Initialize ONNX Runtime session
ort_session = ort.InferenceSession(
  os.getenv("MODEL_PATH", "bert_moe.onnx"),
  providers=["CUDAExecutionProvider", "CPUEExecutionProvider"]
)

app = FastAPIT(title="BERT-MOE Model Server")

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

class PredictionRequest(BaseModel):
  input_ids: list[int]
  attention_mask: list[int]
  token_type_ids: Optional[list[int]] = None

class PredictionResponse(BaseModel):
  predictions: list[float]
  model: str = "bert-moe-v1"

@app.on_event("startup")
async def start_event():
  logger.info("Model server starting up...")
  # Warm up the model
  dummy_input = {
    "input_ids": [0] * 128,
    "attention_mask": [1] * 128,
    "token_type_ids": [0] * 128
  }
  _ = predict(dummy_input)
  logger.info("Model warmed up and ready for inference")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
  try:
    # Convert input to numpy arrays
    input_ids = np.array(request.input_ids, dtype=np.int64)
    attention_mask = np.array(request.attention_mask, dtype=np.int64)
    token_type_ids = np.array(
      request.token_type_ids or [0] * len(request.input_ids),
      dtype=np.int64
    )

    # Ensure proper shape
    if len(input_ids.shape) == 1:
      input_ids = input_ids.reshape(1, -1)
      attention_mask = attention_mask.reshape(1, -1)
      token_type_ids = token_type_ids.reshape(1, -1)

    outputs = ort_session.run(
      None,
      {
        "input_ids": Input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids
      }
    )
    # Process output (adjust based on the model's output)
    predictions = outputs[0].tolist()[0]

    return PredictionResponse(predictions=predictions)

  except Exception as e:
    logger.error(f"Prediction failed: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
  return {"status": "healthy"}?

@app.get("/model")
async def model_info():
  return {
    "name": "bert-moe",
    "version": "1.0.0",
    "framework": "ONNX Runtime"
  }
