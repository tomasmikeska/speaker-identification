import os
from gmm_train import load_models, get_gmm_path, predict
from dataset import get_speakers
from utils import file_listing, dir_listing, relative_path, get_file_name


EVAL_DIR_PATH = relative_path('../data/eval/')


if __name__ == '__main__':
    speakers = get_speakers()
    gmm_models = load_models(speakers)

    for file in file_listing(EVAL_DIR_PATH, 'wav'):
        speaker = predict(gmm_models, file)
        print('%s -> %s' % (get_file_name(file), speaker))
