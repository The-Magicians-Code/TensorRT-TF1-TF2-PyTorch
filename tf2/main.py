import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as Net
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import os
import time

# Optional image to test model prediction.
img_path = './data/elephant.jpg'
model_path = './model'

os.makedirs(model_path, exist_ok=True)
# Path to save the model h5 file.
model_fname = os.path.join(model_path, 'model.h5')

os.makedirs(model_path, exist_ok=True)

img_height = 224

model = Net(weights='imagenet')

def predict(benchmark=False):
    # Load the image for prediction.
    img = image.load_img(img_path, target_size=(img_height, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

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
        print('average(sec):{},fps:{}'.format(mean_delta,fps))

predict(True)
# Save the h5 file to path specified.
# model.save(model_fname)
#model.save("./model_pb")