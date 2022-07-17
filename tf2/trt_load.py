import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import convert_to_constants
import numpy as np

saved_model_loaded = tf.saved_model.load(
    "./model_trt_16", tags=[tag_constants.SERVING])
graph_func = saved_model_loaded.signatures[
    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
frozen_func = convert_to_constants.convert_variables_to_constants_v2(
    graph_func)

BATCH_SIZE = 1

def inputs():
    input_ = np.random.normal(size=(BATCH_SIZE, 224, 224, 3)).astype(np.float32)
    yield [input_]

input_fn = inputs
inp = tf.convert_to_tensor(
    next(input_fn())[0]
)


# Benchmarking throughput
import time
import numpy as np
N_warmup_run = 50
N_run = 1000
elapsed_time = []
for i in range(N_warmup_run):
    preds = frozen_func(inp)
for i in range(N_run):
    start_time = time.time()
    preds = frozen_func(inp)
    end_time = time.time()
    elapsed_time = np.append(elapsed_time, end_time - start_time)
    if i % 50 == 0:
        print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-50:].mean()) * 1000))
print('Throughput: {:.0f} images/s'.format(N_run * BATCH_SIZE / elapsed_time.sum()))
