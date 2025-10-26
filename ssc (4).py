import scipy.io.wavfile as wav
from python_speech_features import ssc
import os
import numpy as np
import pandas as pd
import re

# SVM model libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC




input_folder = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/swallowing_segments"

labelling_csv = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/Swallowing1.csv"

labels_df = pd.read_csv(labelling_csv)
labels_df.columns = labels_df.columns.str.strip()

def aggregation_calculations(output):
    mean = np.mean(output, axis=0)
    std = np.std(output, axis=0)
    return np.concatenate([mean, std])

def sort_key(filename):
    match = re.match(r"P(\d+)_(\d+)\.wav", filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    else:
        return (float('inf'), float('inf'))

# Preprocess aspirating column to dict: participant -> set of swallowing numbers
aspirating_dict = {}
for idx, row in labels_df.iterrows():
    participant = str(row['Participant']).strip()  # Convert to string and strip spaces
    aspirating_value = row['Aspirating']

    # Handle NaN or non-string values safely
    if isinstance(aspirating_value, str):
        aspirating_str = aspirating_value.strip()
    elif pd.isna(aspirating_value):
        aspirating_str = ""
    else:
        aspirating_str = str(aspirating_value).strip()

    if aspirating_str:
        aspirating_numbers = set(int(x.strip()) for x in aspirating_str.split(','))
    else:
        aspirating_numbers = set()

    aspirating_dict[participant] = aspirating_numbers

wav_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.wav')]
sorted_files = sorted(wav_files, key=sort_key)

features_list = []
labels_list = []

for file in sorted_files:
    match = re.match(r"P(\d+)_(\d+)\.wav", file)
    if not match:
        print(f"Filename {file} does not match expected pattern, skipping.")
        continue

    participant_id = f"P{int(match.group(1))}"
    swallow_num = int(match.group(2))

    label_set = aspirating_dict.get(participant_id, set())
    label = 1 if swallow_num in label_set else 0

    try:
        fs, audio = wav.read(os.path.join(input_folder, file))
        if audio.ndim > 1:
            audio = audio[:, 0]
        if len(audio) == 0:
            print(f"{file} does not have audio")
            continue

        output = ssc(audio, samplerate=fs, winlen=0.025, winstep=0.0125,
                     nfilt=26, nfft=2048, lowfreq=0, highfreq=fs/2)
        agg_features = aggregation_calculations(output)

        print(f"Processed {file} - Participant: {participant_id}, Swallow #: {swallow_num}, Label: {label}")

        # Collect features and labels for later use
        features_list.append(agg_features)
        labels_list.append(label)

        
    except Exception as e:
        print(f"Error processing {file}: {e}")

# Convert lists to numpy arrays for ML
X = np.array(features_list)
y = np.array(labels_list)

print(f"Extracted features and labels for {len(X)} audio files.")


#  TRAINING THE SVM MODEL #

# splitting data into training and testing set
# stratisfy helps to preserve the proportion when splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# scaling the feature
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# training the svm classifier 
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)


