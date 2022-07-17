import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Define conversion parameters
precision_mode = "FP16" # FP32, FP16, UINT8 - UINT8 requires calibration, not configured yet
trt_engine_graph_name = f"./trt_engine_{precision_mode}" # TensorRT model path
model_path_pb = "./model_pb" # Frozen model path

# Initialise and set conversion parameters
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(max_workspace_size_bytes=1<<25)
conversion_params = conversion_params._replace(precision_mode=precision_mode)
print(conversion_params)

# Initialise the converter
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=model_path_pb,
    conversion_params=conversion_params
)
converter.convert()

# Initialise a randomised input with corresponding input sizes (this is needed, since we know the
# model inputs which enables the engine to select the best operators which in turn increase the
# inference speed and performance)
def inputs():
    BATCH_SIZE = 1
    input_ = np.random.normal(size=(BATCH_SIZE, 224, 224, 3)).astype(np.float32)
    yield [input_]

# Build and save the TensorRT model
converter.build(input_fn = inputs)
converter.save(trt_engine_graph_name)
print("Model saved to:", trt_engine_graph_name)