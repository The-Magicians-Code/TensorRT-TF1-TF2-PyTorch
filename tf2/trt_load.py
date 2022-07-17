import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import convert_to_constants
import numpy as np

# TensorRT saved model
engine_name = "./trt_engine_FP16"

# Load it with constant tags
saved_model_loaded = tf.saved_model.load(engine_name, tags=[tag_constants.SERVING])

# Generate graph with signatures
graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

# Convert engine constants to variables for inference
frozen_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)

# Self explanatory
BATCH_SIZE = 1

# Generate a random input image with correct parameters
def inputs():
    input_ = np.random.normal(size=(BATCH_SIZE, 224, 224, 3)).astype(np.float32)
    yield [input_]

# Convert it to tensor
input_fn = inputs
inp = tf.convert_to_tensor(
    next(input_fn())[0]
)

# Benchmark the model throughput
import time
import numpy as np
warmups = 50
test_runs = 1000
elapsed_time = []
for i in range(warmups):
    preds = frozen_func(inp)
for i in range(test_runs):
    start_time = time.time()
    preds = frozen_func(inp)
    end_time = time.time()
    elapsed_time = np.append(elapsed_time, end_time - start_time)
    if i % 50 == 0:
        print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-50:].mean()) * 1000))
print('Throughput: {:.0f} images/s'.format(test_runs * BATCH_SIZE / elapsed_time.sum()))
