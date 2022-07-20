import argparse
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Init parser
parser = argparse.ArgumentParser()
parser.add_argument("--precision_mode", 
    type=str,
    choices=["FP32", "FP16"], 
    required=True) # Uint8 Soon_tm
parser.add_argument("--input_model", 
    type=str,
    help="Keras .h5 file or SavedModel .pb, model.h5 or folder consisting model.pb for example", 
    required=True)
parser.add_argument("--batch_size", 
    type=int,
    help="Batch size for configuring TensorRT version of the model", 
    default=1)
parser.add_argument("--save_frozen_dir", 
    type=str,
    help="Saves Keras .h5 model as frozen .pb model to a specified " \
    "directory, can be fed later manually to the TensorRT engine or" \
    " launched to mobile environments", 
    default="./")
parser.add_argument("--trt_engine_dir", 
    type=str,
    help="Saves built TensorRT engine to a specified directory", 
    default="./")

args = parser.parse_args()

def tf1_engine(model_fname=args.input_model,
               save_pb_dir=args.save_frozen_dir,
               trt_engine_graph_dir=args.trt_engine_dir,
               precision_mode=args.precision_mode,
               batch_size=args.batch_size):
    """This is for TensorFlow 1, once you'll find out what numbers are

    Args:
        model_fname (str, required): Model name.
        save_pb_dir (str, required): Frozen graph path.
        trt_engine_graph_dir (str, required): Same, but for TensorRT engine.
        precision_mode (str, required): FP16, FP32, UINT8 -> Soon_tm.
        batch_size (int, optional): Title.

    Returns:
        TensorRT engine: Does what the title says
    """

    from tensorflow.python.framework import graph_io  # Graph generator
    from tensorflow.keras.models import load_model  # For loading the Keras model
    from tensorflow.keras.backend import get_session  # Session as in TensorFlow 1

    trt_engine_graph_name = f"{model_fname[:-3]}_trt_engine_{precision_mode}.pb"

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
    with open(f"{model_fname[:-3]}_data.txt", "w") as data:
        data.write(f"{input_names[0]},{output_names[0]}\n")

    # Generate the frozen graph
    frozen_graph = freeze_graph(session.graph, 
                                session, 
                                output_names, 
                                save_pb_dir=save_pb_dir, 
                                save_pb_name=f"frozen_{model_fname[:-3]}.pb")

    # Generate TensorRT graph
    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output_names,
        max_batch_size=batch_size,
        max_workspace_size_bytes=1<<25,
        precision_mode=precision_mode,
        minimum_segment_size=50
    )

    # Save the generated TensorRT engine graph
    graph_io.write_graph(trt_graph, trt_engine_graph_dir,
                        trt_engine_graph_name, as_text=False)

    # Clear any previous session.
    tf.keras.backend.clear_session()
        
def tf2_engine(model_fname=args.input_model,
               precision_mode=args.precision_mode,
               batch_size=args.batch_size):
    """This is for TensorFlow 2, when you learn to count to 1 and further

    Args:
        model_fname (str, required): Model name.
        precision_mode (str, required): Precision mode, can't you read?
        batch_size (int, optional): I'm asking once again.

    Yields:
        TensorRT engine: Does what the title says
    """
    import numpy as np  # For computing the last five digits of Pi

    if model_fname.endswith(".h5"): # If it's a Keras model
        model = tf.keras.models.load_model(model_fname)
        model.save(f"{model_fname[:-3]}_saved_model") # Save it as ProtoBuf model
        model_path_pb = f"{model_fname[:-3]}_saved_model"
    else:
        model_path_pb = model_fname[:-3]  # Don't do anything if you have the ProtoBuf model

    # Self explanatory
    trt_engine_graph_name = f"{model_fname[:-3]}_trt_engine_{precision_mode}.pb"

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

    # Dynamical input size selection, witty and posh looking, unlike you
    print([input for input in model.inputs], "\nSelecting the first input params:")
    configured_size = model.inputs[0].shape.as_list()
    configured_size[0] = batch_size
    
    # Initialise a randomised input with corresponding input sizes (this is needed, since the
    # model inputs are known, which enables the engine to select the best operators which in 
    # turn increase the inference speed and performance, not that you could ever fathom these terms)
    def inputs():
        input_ = np.random.normal(size=configured_size).astype(np.float32)
        yield [input_]

    # Build and save the TensorRT model, simple, and efficient, just like eating a spoonful of
    # Uranium
    converter.build(input_fn = inputs)
    converter.save(trt_engine_graph_name)
    print("Model saved to:", trt_engine_graph_name)

print(f"Using TensorFlow version: {tf.__version__}")

input_model = args.input_model
if input_model.endswith(".pb") and tf.__version__.startswith("2"):
    print("ProtoBuf model, no freezing needed")
elif input_model.endswith(".h5"):
    print("Keras model, creating a frozen graph (ProtoBuf) model")
else:
    raise ValueError("Unknown format, .h5 (TensorFlow 1 and 2) or .pb " \
    "(TensorFlow 2) accepted")

print("Proceeding to TensorRT engine building operations")

# Choosing the correct engine
tf1_engine() if tf.__version__.startswith("1") else tf2_engine()
