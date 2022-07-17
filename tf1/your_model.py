import tensorflow as tf
# Uncomment the next two lines in case you're that one smartass who somehow tries to run it on tf2
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()    # read the previous comment again, dumbass
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as Net
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import time

# Optional image to test model accuracy
img_path = "./data/elephant.jpg"
model_ = "./model/model.h5"

# Parameters
predictions = True
benchmark = True

model = Net(weights="imagenet")

def predict(benchmark=False):
    """Tests model accuracy (and speed if in benchmark mode)

    Args:
        benchmark (bool, optional): Enables inference speed test. Defaults to False.
    """
    # Load the image for prediction.
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    print("Predicted:", decode_predictions(preds, top=3)[0])

    if benchmark:
        # Inference benchmark
        times = []
        for i in range(20):
            start_time = time.time()
            preds = model.predict(x)
            delta = (time.time() - start_time)
            times.append(delta)
        mean_delta = np.array(times).mean()
        fps = 1/mean_delta
        print("average(sec):{},fps:{}".format(mean_delta,fps))

if predictions:
    predict(benchmark=benchmark)

# Save the model file to path specified
model.save(model_)

# Clear all sessions
tf.keras.backend.clear_session()