import scipy.io.wavfile as wav
from python_speech_features import ssc
import os
import numpy as np

input_folder = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/swallowing_segments"

def aggregation_calculations(output):
    mean = np.mean(output, axis = 0)
    std = np.std(output, axis = 0)
    return np.concatenate([mean, std])



# going through each wav file in the folder
for file in os.listdir(input_folder):

    # if it is not a wav file, skip the file
    if not file.lower().endswith('.wav'):
        print(f"Skipping non-wav file: {file}")
        continue  # Skip non-wav files

    try:
        fs, audio = wav.read(os.path.join(input_folder, file))

        # if there is audio in that file
        if len(audio) > 0:
            output = ssc(audio, samplerate=fs, winlen=0.025, winstep=0.0125, nfilt=26, nfft=2048, lowfreq=0, highfreq=fs/2)

            agg_features = aggregation_calculations(output)
            


            # print(output)
            print(f"{file}: SSC feature frames = {output.shape[0]}")
            print(f"{file}: SSC features per frame = {output.shape[1]}")
            print(len(agg_features))
        else:
            print(f"{file} does not have audio")

    except ValueError as ve:
        print(f"ValueError reading {file}: {ve}")
    except Exception as e:
        print(f"Error processing {file}: {e}")

