from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class FruitModel:

    def __init__(self):
        self.model = self.load_model('Fruit_classification_model.h5')

    def load_model(self, model_path):
        return load_model(model_path)

    def show_image(self, image_path):
        image = mpimg.imread(image_path)
        print(image.shape)
        plt.imshow(image)


    def make_predictions(self, image_path):
        self.show_image(image_path)
        image = image_utils.load_img(image_path, target_size=(224, 224))
        image = image_utils.img_to_array(image)
        image = image.reshape(1,224,224,3)
        image = preprocess_input(image)
        preds = self.model.predict(image)
        return preds[0][0]

    def fresh_or_rotten(self, image_path):
        preds = self.make_predictions(image_path)
        if preds <= 0.5:
            print("It's Fresh! eat ahead.")
        else:
            print("It's Rotten, I wont recommend!")