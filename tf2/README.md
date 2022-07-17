# Code related to model quantisation with TensorRT for TensorFlow 2.x

### 1. your_model
Well... as the title suggests
Creates a MobileNetV2 model instance and saves it as a protobuf (.pb) frozen model file

### 2. build_trt
Loads the previously created model, optimises it with TensorRT and saves the
optimised model engine

### 3. load_trt
Loads optimised TensorRT model engine and tests it

### model_data (generated)
Text file which contains model input and output layer info
