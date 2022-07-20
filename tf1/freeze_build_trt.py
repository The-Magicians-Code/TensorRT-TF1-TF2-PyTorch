import tensorflow as tf # By default, as usual
# Uncomment the next two lines in case you're that one smartass who somehow tries to run it on tf2
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()    # read the previous comment again, dumbass
from tensorflow.python.framework import graph_io    # Graph generator
from tensorflow.keras.models import load_model  # For loading the Keras model
from tensorflow.python.compiler.tensorrt import trt_convert as trt  # The TensorRT converter
from tensorflow.keras.backend import get_session  # Session as in TensorFlow 1

# Clear any previous session.
tf.keras.backend.clear_session()

# Define your parameters
save_pb_dir = "./model/"
model_fname = "./model/model.h5"
trt_engine_graph_dir = "./model/"
precision_mode = "FP16" # FP32, FP16, UINT8 - UINT8 requires calibration, not configured yet
trt_engine_graph_name = f"trt_engine_{precision_mode}.pb"

def freeze_graph(graph, session, output, save_pb_dir=".", save_pb_name="frozen_model.pb", 
                 save_pb_as_text=False):
    """Generates a frozen protobuf (.pb) graph from loaded Keras model

    Args:
        graph: Session graph
        session: tf.Session
        output: Model output layer names
        save_pb_dir (str, optional): Frozen graph save directory. Defaults to ".".
        save_pb_name (str, optional): Frozen graph name. Defaults to "frozen_model.pb".
        save_pb_as_text (bool, optional): Save graph as text file. Defaults to False.

    Returns:
        graphdef_frozen: Frozen Keras model graph instance
    """
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0)

# Self explanatory
model = load_model(model_fname)

# Start a session
session = get_session()

# Obtain model inputs and outputs and their respective names
input_names = [t.op.name for t in model.inputs]
output_names = [t.op.name for t in model.outputs]

# Print input and output node names and write them to a text file
print(input_names, output_names)
with open("model_data.txt", "w") as data:
    data.write(f"{input_names[0]},{output_names[0]}\n")

# Generate the frozen graph
frozen_graph = freeze_graph(session.graph, session, output_names, save_pb_dir=save_pb_dir)

# Generate TensorRT graph
trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1<<25,
    precision_mode=precision_mode,
    minimum_segment_size=50
)

# Save the generated TensorRT engine graph
graph_io.write_graph(trt_graph, trt_engine_graph_dir,
                     trt_engine_graph_name, as_text=False)
