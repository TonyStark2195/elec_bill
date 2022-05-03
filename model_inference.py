import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
from tqdm import tqdm


class ModelAPI:
    def __init__(self, model_path=""):
        # load model in memory
        self.model = keras.models.load_model(model_path)
        self.prediction = {
            0: "Vegetated",
            1: "Un-vegetated",
            2: "Water"
        }
        pass

    def inference(self, img_loc="", image_size=(500, 500)):
        """
        Model inference method.
        :param image: Image to be classified
        :return: label and its confidence
        """

        img = keras.preprocessing.image.load_img(
            img_loc, target_size=image_size
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = self.model.predict(img_array)
        prob_vals = predictions[0]

        return max(prob_vals), np.argmax(prob_vals)


if __name__ == '__main__':
    model_path = r"checkpoint\save_at_50.h5"
    model = ModelAPI(model_path)
    img_dir = r"D:\revegetation\landloss 7k\images"  # r"D:\revegetation\landloss_2021"

    # df = pd.read_excel("Cartoscope oil and gas wells vegetation 2022.xlsx")
    df = pd.read_csv("Unplugged Coastal Oil and Gas Wells 2018 Dec.csv")
    pred_class_list = list()
    pred_prob_list = list()

    for serial in tqdm(df['WELL_SERIA'].values.tolist(),
                       desc="Model inference in Progress: "):
        src = glob.glob(img_dir + "\{}*_image.png".format(int(serial)))[0]

        class_prob, class_pred = model.inference(img_loc=src)
        pred_class_list.append(class_pred + 1)
        pred_prob_list.append(class_prob)

    df['predicted labels'] = pred_class_list
    df['predicted probability'] = pred_prob_list

    # df['predicted labels'].replace({1: 2, 2: 1})

    df.to_csv('results_7k.csv')
