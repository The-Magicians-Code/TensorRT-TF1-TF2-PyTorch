import tensorflow as tf
# Uncomment the next two lines in case you're that one smartass who somehow tries to run it on tf2
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()    # read the previous comment again, dumbass
import numpy as np
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Self explanatory
engine_name = "./model/trt_engine_FP16.pb" # And location
# Image to test model prediction
img_path = "./data/elephant.jpg"

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

trt_graph = get_frozen_graph(engine_name)

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name="")

# Get graph input size
for node in trt_graph.node:
    if "input_" in node.name:
        size = node.attr["shape"].shape
        image_size = [size.dim[i].size for i in range(1, 4)]
        break
print("image_size: {}".format(image_size))

# Open the file with previously saved parameters
with open("model_data.txt", "r") as data:
    input_names, output_names = zip(*[item.strip().split(",") for item in data])

input_tensor_name = input_names[0] + ":0"
output_tensor_name = output_names[0] + ":0"

print("input_tensor_name: {}\noutput_tensor_name: {}".format(
    input_tensor_name, output_tensor_name))

# Use the output node name to select the correct tensor
output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)

# Prediction time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load it and preprocess it
img = image.load_img(img_path, target_size=image_size[:2])
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

feed_dict = {
    input_tensor_name: x
}

# Make predictions
preds = tf_sess.run(output_tensor, feed_dict)
print(preds.shape)

# Decode the results into a list of tuples (class, description, probability)
print("Predicted:", decode_predictions(preds, top=3)[0])

# Benchmark
import time
times = []
for i in range(30):
    pred_warmup = tf_sess.run(output_tensor, feed_dict)
for i in range(20):
    start_time = time.time()
    one_prediction = tf_sess.run(output_tensor, feed_dict)
    delta = (time.time() - start_time)
    times.append(delta)
mean_delta = np.array(times).mean()
fps = 1 / mean_delta
print("Average(sec):{:.2f}, fps:{:.2f}".format(mean_delta, fps))

# Free resources
tf_sess.close()