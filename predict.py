import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('pokemon_model.keras')
class_names = [...]  # Preencha com suas classes

def predict_pokemon(img_path):
    img = Image.open(img_path).resize((224, 224))
    img_array = tf.keras.applications.efficientnet.preprocess_input(np.array(img))
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    return class_names[np.argmax(prediction)]

# Teste:
print(predict_pokemon('test_images/aegislash_test.png'))