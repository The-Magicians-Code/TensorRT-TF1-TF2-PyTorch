import os
import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

model_path_trt = "./model_trt_16" # define path to save model in saved model format
model_path_pb = "./model_pb"

os.makedirs(model_path_trt, exist_ok=True)
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(max_workspace_size_bytes=1<<25)
conversion_params = conversion_params._replace(precision_mode="FP16")

print(conversion_params)
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=model_path_pb,
    conversion_params=conversion_params
)
converter.convert()

def inputs():
    BATCH_SIZE = 1
    input_ = np.random.normal(size=(BATCH_SIZE, 224, 224, 3)).astype(np.float32)
    yield [input_]

converter.build(input_fn = inputs)
converter.save(model_path_trt) # we save the converted model
print("trt-model saved to:", model_path_trt)