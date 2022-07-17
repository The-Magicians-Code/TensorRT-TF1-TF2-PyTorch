Code related to model quantisation with TensorRT for TensorFlow 2.x

# your_model
Well... as the title suggests
Creates a MobileNetV2 model instance and saves it as a protobuf (.pb) frozen model file

# build_trt
Loads the previously created model, optimises it with TensorRT and saves the
optimised model engine

# load_trt
Loads optimised TensorRT model engine and tests it

# model_data (generated)
Text file which contains model input and output layer info