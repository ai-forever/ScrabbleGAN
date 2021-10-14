import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import cv2
import os
from collections import Counter

from config import Config


def get_full_img_path(img_path, path_to_csv):
    root_path = os.path.split(path_to_csv)[0]
    return os.path.join(root_path, img_path)


def read_image(img_path, label_len, img_h=32, char_w=16, channels=3):
    valid_img = True
    if channels == 3:
        img = cv2.imread(img_path)
    else:
        img = cv2.imread(img_path, 0)
        img = np.expand_dims(img, -1)
    try:
        curr_h, curr_w, _ = img.shape
        modified_w = int(curr_w * (img_h / curr_h))

        # Remove outliers
        if ((modified_w / label_len) < (char_w / 3)) | ((modified_w / label_len) > (3 * char_w)):
            valid_img = False
        else:
            # Resize image so height = img_h and width = char_w * label_len
            img_w = label_len * char_w
            img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)

    except AttributeError:
        valid_img = False

    return img, valid_img


def read_data(config, args):
    """
    Saves dictionary of preprocessed images and labels for the required partition
    """
    # read csv data
    data = []
    for csv_path in args.data_csv_path:
        print(f"processed: {csv_path}")
        csv_data = pd.read_csv(csv_path)
        csv_data['filename'] = csv_data['filename'].apply(
            get_full_img_path, path_to_csv=csv_path)
        data.append(csv_data)
    data = pd.concat(data, ignore_index=True)

    # Get list of unique characters and create dictionary for mapping them to integer
    # chars = np.unique(np.concatenate(
    #     [[char for char in str(w_i)] for w_i in data['text'].values]))
    chars = []
    char2count = Counter(np.concatenate([[char for char in str(w_i)]
                                         for w_i in data['text'].values]))
    for char, count in char2count.items():
        if count > args.remove_rare:
            chars.append(char)
    if args.remove_letters:
        chars = [char for char in chars if not char.isalpha()]

    char_map = {value: idx + 1 for (idx, value) in enumerate(chars)}
    char_map['<BLANK>'] = 0
    num_chars = len(char_map.keys())

    word_data = {}
    for idx, word in enumerate(data['text'].values):
        word = str(word)
        if set(word).union(set(chars)) == set(chars):
            img_path = data['filename'][idx]
            img, valid_img = read_image(img_path, len(word), config.img_h, config.char_w, channels=config.channels)
            if valid_img:
                word_data[idx] = [[char_map[char] for char in word], img]

    print(f'Number of images = {len(word_data)}')
    print(f'Number of unique characters = {num_chars}')

    # Save the data
    with open(args.output_pkl_name, 'wb') as f:
        pkl.dump({'word_data': word_data,
                  'char_map': char_map,
                  'num_chars': num_chars}, f, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv_path", action='append', help="Path to the "
                        "dataset annotation csv. Can be passed multiple times",
                        required=True)
    parser.add_argument("--output_pkl_name", required=True, type=str,
                        help="Name of the output pickle file")
    parser.add_argument("--remove_letters", action='store_true',
                        help="To remove letters samples from data.")
    parser.add_argument("--remove_rare", type=int, default=0,
                        help="The minimum number of times a char appears in "
                        "the data to remove the char from the dataset. "
                        "Default is 0 which mean no chars will be removed.")
    args = parser.parse_args()

    read_data(Config, args)
