import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as Net
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import time

# Define parameters
img_path = "./data/elephant.jpg"
model_path = "./model_pb"

prediction = True
benchmark = True

# Initialise the model
model = Net(weights="imagenet")

def predict(benchmark=False):
    # Load the image for prediction
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # Decode the results into a list of tuples (class, description, probability)
    print("Predicted:", decode_predictions(preds, top=3)[0])

    if benchmark:
        # Speed benchmark
        times = []
        for i in range(20):
            start_time = time.time()
            preds = model.predict(x)
            delta = (time.time() - start_time)
            times.append(delta)
        mean_delta = np.array(times).mean()
        fps = 1/mean_delta
        print("Average(sec):{}, fps:{}".format(mean_delta,fps))

if predict:
    predict(benchmark=benchmark)

# Save the model protobuf file (.pb frozen model) to the path specified
model.save(model_path)