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
