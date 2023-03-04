import os
import glob

import cv2
from torch.utils.data import Dataset


class customData(Dataset):
    def __init__(self, path, transform=None):
        self.all_data = glob.glob(os.path.join(path, "*", "*.png"))
        self.transform = transform

        # label dict
        self.label_dict = {}
        for i, (label) in enumerate(os.listdir("data/train")):
            self.label_dict[label] = int(i)

    def __getitem__(self, item):
        # image path
        image_path = self.all_data[item]

        # image read
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # label
        label_temp = image_path.split("/")[2]
        label_temp = label_temp.split("\\")[1]
        label = self.label_dict[label_temp]

        # transform
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label

    def __len__(self):
        return len(self.all_data)

# test = customData("./data/tran/", transform=None)
# for i in test:
#     print(i)
