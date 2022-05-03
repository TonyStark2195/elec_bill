import shutil
import os
import glob
from tqdm import tqdm
import pandas as pd


class DataPrep:
    def __init__(self):
        self.cur_img_dir = r"D:\revegetation\landloss_2021"
        self.des_img_dir = r"landloss_model_training_images"
        self.label = {
            1: "Vegetated",
            2: "Un-vegetated",
            3: "Water"
        }

    def sort(self):
        os.makedirs(self.des_img_dir, exist_ok=True)
        df = pd.read_excel("Cartoscope oil and gas wells vegetation 2022.xlsx")
        unique_labels = df['designationspe'].unique()
        for un_lab in unique_labels:
            os.makedirs(os.path.join(self.des_img_dir, self.label[int(un_lab)]), exist_ok=True)

        for serial, label in tqdm(df[['WELL_SERIA', 'designationspe']].values.tolist(),
                                  desc="Data moving in Progress: "):
            src = glob.glob(self.cur_img_dir + "\*_{}*_combined.jpg".format(int(serial)))[0]
            if len(src) > 0:
                dest = self.des_img_dir + "\\" + str(self.label[int(label)]) + "\{}.jpg".format(str(serial))
                shutil.copy(src, dest)


if __name__ == '__main__':
    data = DataPrep()
    data.sort()
