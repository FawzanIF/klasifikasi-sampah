import numpy as np
import tensorflow as tf

from flask import Flask, request
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/klasifikasi-sampah", methods = ['POST'])
def klasifikasi_sampah_classifier ():
    
    # ambil gambar yang dikirim pas request
    image_request = request.files['image']
    
    # konversi gambar menjadi array
    image_pil = Image.open(image_request)

    # ngeresize gambar
    expected_size = (224, 224)
    resized_image_pil = image_pil.resize(expected_size)

    # generate array dengan numpy
    image_array = np.array(resized_image_pil)
    rescaled_image_array = image_array/255.
    batched_rescaled_image_array = np.array([rescaled_image_array])
    print(batched_rescaled_image_array.shape)

    
    # load model
    loaded_model = tf.keras.models.load_model('model_sampah_skenario1.h5', compile=False)

    try:
        result = loaded_model.predict(batched_rescaled_image_array)
    except:
        return "Error: Gambar yang diunggah tidak dikenali sebagai sampah yang dikenali oleh model."

    return get_formated_predict_result(result)

def get_formated_predict_result(predict_result) :
    class_indices = {'Anorganik': 0, 'B3': 1, 'Kertas': 2, 'Organik': 3, 'Residu': 4, }
    inverted_class_indices = {}

    for key in class_indices:
        class_indices_key = key
        class_indices_value = class_indices[key]

        inverted_class_indices[class_indices_value] = class_indices_key

    processed_predict_result = predict_result[0]
    maxIndex = 0
    maxValue = 0

    for index in range(len(processed_predict_result)):
        if processed_predict_result[index] > maxValue:
            maxValue = processed_predict_result[index]
            maxIndex = index

    return inverted_class_indices[maxIndex]

if __name__ == "__main__":
    app.run()