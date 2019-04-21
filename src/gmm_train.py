import os
import pickle
from numpy import inf
from sklearn.mixture import GaussianMixture as GMM
from dataset import load_local_dataset, read_wav
from utils import file_listing, dir_listing, last_component, relative_path

# Hyperparameters
MIXTURE_COMPONENTS = 24
MAX_ITERS = 300
NUM_INITS = 10
MODEL_PERSIST_PATH = relative_path('../model/gmm/')

def get_gmm_path(speaker):
    return MODEL_PERSIST_PATH + 'speaker%s_%scomps_%smaxiter_%sninit.gmm' % (speaker, MIXTURE_COMPONENTS, MAX_ITERS, NUM_INITS)

def load_gmm(speaker):
    with open(get_gmm_path(speaker), 'rb') as f:
        gmm = pickle.load(f)
    return gmm

def load_models(speakers):
    if len(file_listing(MODEL_PERSIST_PATH, 'gmm')) > 0:
        print('Loading saved GMM models from file')
        return { speaker: load_gmm(speaker) for speaker in speakers }
    else:
        print('GMMs need to be trained first')
        exit(1)

def save_gmms(gmms):
    for speaker, gmm in gmms.items():
        with open(get_gmm_path(speaker), 'wb+') as f:
            pickle.dump(gmm, f)

def train_gmms(gmms):
    for speaker, recordings in speaker_recordings.items():
        print('Training GMM for speaker %s' % speaker)
        gmms[speaker].fit(recordings)

def init_gmm():
    return GMM(n_components=MIXTURE_COMPONENTS,
               max_iter=MAX_ITERS,
               n_init=NUM_INITS,
               covariance_type='diag')

def predict_speaker(gmms, X):
    top_score = -inf
    top_speaker = 0
    for speaker, gmm in gmms.items():
        score = gmm.score(X)
        if score > top_score:
            top_score = score
            top_speaker = speaker
    return top_speaker

def predict(gmms, file):
    return predict_speaker(gmms, read_wav(file))


if __name__ == '__main__':

    speaker_recordings, speakers = load_local_dataset()
    gmm_models = { speaker: init_gmm() for speaker in set(speakers) }
    train_gmms(gmm_models)
    save_gmms(gmm_models)

    # Calculate precision
    total = 0
    correct = 0
    wrong_files = {}

    for dir_path in dir_listing(relative_path('../data/dev/')):
        for file in file_listing(dir_path, 'wav'):
            print('Predicting file %s (%s/%s)' % (file, correct, total + 1))
            speaker = predict(gmm_models, file)
            if speaker == last_component(dir_path):
                correct += 1
            else:
                wrong_files[file] = (speaker, last_component(dir_path))
            total += 1

    print('Incorrectly classified')
    for file, val in wrong_files.items():
        print('%s is %s but was classfied as %s' % (file, val[1], val[0]))

    print('Got %s correct out of %s' % (correct, total))
    print('-> %s percent accuracy' % ((correct / total) * 100))
