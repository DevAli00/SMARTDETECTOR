import json
import os
import numpy as np
import math
import librosa
DATASET_PATH = "" #Your dataset trajectory
JSON_PATH = "data.json"

Sampling_rate = 44100 
Duration = 7
Sample_per_track = Sampling_rate * Duration


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=1024, hop_length=512, num_segments=1):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    Sample_per_segment = int(Sample_per_track / num_segments)
    mfcc_vectors_number = math.ceil(Sample_per_segment / hop_length)

    for i, (directory_path, directory_name, files_name) in enumerate(os.walk(dataset_path)):

        if directory_path != dataset_path:

            composants_chemin_répertoire = directory_path.split("/")
            label = composants_chemin_répertoire[-1]
            data["mapping"].append(label)
            print("\nProcessing of {}".format(label))

            for file_name in files_name:
                file_path = os.path.join(directory_path, file_name)
                signal, sr = librosa.load(file_path, sr=Sampling_rate)

                for s in range(num_segments):
                    beginning = Sample_per_segment * s
                    end = beginning + Sample_per_segment
                    

                    mfcc = librosa.feature.mfcc(y=signal[beginning:end],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    if len(mfcc) == mfcc_vectors_number:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment : {}".format(file_path, s))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

save_mfcc(DATASET_PATH, JSON_PATH, num_segments=3)
