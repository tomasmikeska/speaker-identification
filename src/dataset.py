import numpy as np
import pickle
from scipy.io import wavfile
from feature_extraction import extract_features, OUTPUT_DIM
from utils import file_listing, dir_listing, last_component, relative_path, file_exists


DATASET_TRAIN_PATH = relative_path('../data/train/')
TRAIN_PERSIST_PATH = relative_path('../data/train.npy')


def read_wav(filepath):
    freq, audio = wavfile.read(filepath)
    return extract_features(audio, freq)


def read_dataset_dir(base_dir):
    dirs = dir_listing(base_dir)
    X = {last_component(dir): np.empty(OUTPUT_DIM) for dir in dirs}

    for dir_path in dirs:
        for file in file_listing(dir_path, 'wav'):
            speaker = last_component(dir_path)
            audio_np = read_wav(file)
            X[speaker] = np.vstack((X[speaker], audio_np))

    return X


def get_speakers(base_dir=DATASET_TRAIN_PATH):
    return set(map(lambda dir_path: last_component(dir_path), dir_listing(base_dir)))


def load_data():
    with open(TRAIN_PERSIST_PATH, 'rb') as f:
        data = pickle.load(f)
    return data


def save_data(train):
    with open(TRAIN_PERSIST_PATH, 'wb+') as f:
        pickle.dump(train, f)


def load_local_dataset():
    if file_exists(TRAIN_PERSIST_PATH):
        print('Loading dataset from npy file')
        return load_data(), get_speakers()
    else:
        print('Reading and tranforming dataset')
        train = read_dataset_dir(DATASET_TRAIN_PATH)
        save_data(train)
        return train, get_speakers()
