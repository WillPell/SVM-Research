import numpy as np
import h5py as h5
import scipy.io.wavfile as wf
import pandas as pd
import os

# This code helps to read the participant audio files and swallow event labels from CSV
# Convers various timestamp formats into sample indices relative to audio data
# Creates binary label arrays marking where swallows occur in the audio
# Saves raw audio and label data efficiently in an HDF5 file for downstream processing or machine learning


rawSave = 1  # set to save raw data again

def convertTimestamp(timestamp, Fs):
    if isinstance(timestamp, float) or isinstance(timestamp, int):
        time_in_seconds = float(timestamp)
    else:
        # removing the extra literal string around the column names
        timestamp = str(timestamp).strip().strip("'").strip('"')

        # Fix common typo: replace ';' with ':' (typo)
        timestamp = timestamp.replace(';', ':')
        
        try:
            if '.' in timestamp and timestamp.count('.') == 2 and ':' not in timestamp:
                # Format mm.ss.ms (some recordings were done using this format)
                parts = timestamp.split('.')
                if len(parts) == 3:
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    milliseconds = float(parts[2])
                    time_in_seconds = minutes * 60 + seconds + milliseconds / 1000
                else:
                    raise ValueError
            elif ':' in timestamp:
                # Format mm:ss.sss (some recordings were done using this format)
                parts = timestamp.split(':')
                if len(parts) == 2:
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    time_in_seconds = minutes * 60 + seconds
                else:
                    raise ValueError
            else:
                # Just seconds as float string
                time_in_seconds = float(timestamp)
        except ValueError:
            print(f"Warning: Unexpected timestamp format '{timestamp}'. Treating as 0 seconds.")
            time_in_seconds = 0.0

    # getting where the swallowing occur in the sample index
    sample_index = int(round(time_in_seconds * Fs))

    # returning time when it happens and the sample when it happens 
    return time_in_seconds, sample_index


if __name__ == "__main__":
    dataPath = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/mono_audio"
    labelPath = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/cfe/study 2_clean_swallow sounds.csv"

    labelcsv = pd.read_csv(labelPath)
    labelcsv.columns = [col.strip().strip("'").strip('"') for col in labelcsv.columns]
    print([f"'{col}'" for col in labelcsv.columns.tolist()])

    with h5.File("labels.h5", "w") as v5:
        for idx, row in labelcsv.iterrows():
            p = row['Participant']  # e.g., "P9"
            filename = f"{p}_mono.wav"
            filepath = os.path.join(dataPath, filename)

            print("Reading ", filename)  # visualising in terminal

            try:
                (Fs, x) = wf.read(filepath)
            except (FileNotFoundError, OSError) as e:
                print(f"Warning: Audio file not found or unreadable: {filepath}. Skipping participant {p}.")
                continue  # For now as we have duplicates in the wav file

            Nx = len(x)  # total number of audio samples in the loaded audio file
            print("Number of samples: ", Nx)

            if rawSave == 1:
                dset = v5.create_dataset(f"raw/{p}_1", (Nx,), dtype='int16', fletcher32=True)
                if x.ndim > 1:
                    dset[...] = x[:, 0]  # stereo: take first channel
                else:
                    dset[...] = x   # mono: save as is (getting the amplitude of each frame and store it)

            numSwallows = row['Total swallows']
            print("\tNumber of swallows = ", numSwallows)

            startTime = np.zeros(int(numSwallows), dtype=float)
            stopTime = np.zeros(int(numSwallows), dtype=float)
            startSample = np.zeros(int(numSwallows), dtype=int)
            stopSample = np.zeros(int(numSwallows), dtype=int)

            classLabels = np.zeros(Nx)

            for swallow in range(int(numSwallows)):
                column1 = f"Swallow {swallow + 1} Start"
                column2 = f"Swallow {swallow + 1} Stop"

                startStamp = row[column1]
                stopStamp = row[column2]
                (startTime[swallow], startSample[swallow]) = convertTimestamp(startStamp, Fs)

                # assigning the time and index of swallowing sound
                (stopTime[swallow], stopSample[swallow]) = convertTimestamp(stopStamp, Fs)

                classLabels[startSample[swallow]:stopSample[swallow]] = 1

            dset = v5.create_dataset(f"labels/{p}/start/times", data=startTime, fletcher32=True)
            dset = v5.create_dataset(f"labels/{p}/stop/times", data=stopTime, fletcher32=True)
            dset = v5.create_dataset(f"labels/{p}/start/samples", data=startSample, fletcher32=True)
            dset = v5.create_dataset(f"labels/{p}/stop/samples", data=stopSample, fletcher32=True)

            dset = v5.create_dataset(f"labels/{p}/class", data=classLabels, dtype="int8", fletcher32=True)


