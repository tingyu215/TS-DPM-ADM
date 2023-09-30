import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


def imgs_to_npz():
    npz = []

    for img in tqdm(os.listdir("/export/home1/NoCsBack/working/tingyu/adm/lsun_tower_train")):
        img_arr = cv2.imread("/export/home1/NoCsBack/working/tingyu/adm/lsun_tower_train/" + img)
        npz.append(img_arr)

    output_npz = np.array(npz)
    np.savez('/export/home1/NoCsBack/working/tingyu/adm/lsun_tower_train/lsun_tower_train.npz', output_npz)
    print(f"{output_npz.shape} size array saved into lsun_tower_train.npz")  # (708264, 64, 64, 3)

if __name__ == "__main__":
    imgs_to_npz()