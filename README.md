# Speaker Identification

This project trains a GMM (Gaussian Mixture Model) to identify speaker. Model is trained and evaluated on single-speaker recordings in (ideally) controlled environments. Multiple preprocessing and feature extraction techniques are used.

#### Requirements

- Python 3.x
- pip

### Installation and setup

Install pip packages using
```
$ pip install -r requirements.txt
```

Then put your local dataset into `data` file in project root. Your dataset should be split into 3 subfolders, `train`, `test` and `eval`. Train and test folders should have a subfolder for each speaker with his/hers id as folder name containing `{whatever}.wav` files with recordings. Eval folder is flat and should contain wav files directly.

Example:
```
.
+-- data
|   +-- train
|       +-- speaker1
|           ├── a1.wav
|           └── a2.wav
|       +-- speaker2
|           ├── b1.wav
|           └── b2.wav
|       +-- ...
|   +-- test
|       +-- speaker1
|           ├── a1.wav
|           └── a2.wav
|       +-- speaker2
|           ├── b1.wav
|           └── b2.wav
|       +-- ...
|   +-- eval
|       ├── unknown1.wav
|       └── unknown2.wav
```

#### Usage

Train GMM model using command
```
$ python src/gmm_train.py
```

After training the gmm params are saved to `model/gmm/` folder and processed training dataset to `data/train.npy`.

Evaluate using command

```
$ python src/gmm_eval.py
```

🎉 🎉
