import onnx
from onnxruntime_tools import optimizer

def optimize_model(input_path, output_path):
    model = onnx.load(input_path)
    optimized_model = optimizer.optimize(model)
    onnx.save(optimized_model, output_path)

if __name__ == "__main__":
    optimize_model("model.onnx", "bert_moe.onnx")

