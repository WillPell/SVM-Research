import scipy.io.wavfile as wav
from python_speech_features import ssc
import os
import numpy as np
import pandas as pd
import re

# Import SMOTE from imbalanced-learn 
from imblearn.over_sampling import SMOTE

# SVM model libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Input folder containing swallowing audio segments
input_folder = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/swallowing_segments_cfe"

# # CSV file containing aspiration labels
# labelling_csv = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/paed_vfss/vinod_edited_spreadsheet-VFSSdatset-22Aug2022.csv"

# CSV file for the CFE dataset
labelling_csv = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/Swallowing1.csv"

# Load CSV and clean column names
labels_df = pd.read_csv(labelling_csv)
labels_df.columns = labels_df.columns.str.strip()

def aggregation_calculations(output):
    """Aggregate frame-wise features by mean and std deviation."""
    mean = np.mean(output, axis=0)
    std = np.std(output, axis=0)
    return np.concatenate([mean, std])

def sort_key(filename):
    """Sort files by participant and swallow number extracted from filename without 'P' prefix."""
    match = re.match(r"(\d+)_(\d+)\.wav", filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    else:
        return (float('inf'), float('inf'))

# Build a dictionary mapping participant -> set of aspirated swallow numbers

aspirating_dict = {}
for idx, row in labels_df.iterrows():
    participant = str(row['Participant']).strip()
    aspirating_flag = row.get('Aspirating', 0)
    which_swallows = row.get('Which Swallows', "")

    print(f"Participant: {participant}, Aspirating flag: {aspirating_flag}, Which Swallows: {which_swallows}")

    if aspirating_flag == 1:
        # Parse aspirated swallow numbers robustly
        aspirating_numbers = set()
        if pd.notna(which_swallows):
            # Convert to string for safe splitting
            which_swallows_str = str(which_swallows).strip()
            if which_swallows_str:
                for x in which_swallows_str.split(','):
                    x = x.strip()
                    try:
                        aspirating_numbers.add(int(float(x)))
                    except ValueError:
                        # Ignore invalid entries
                        pass
    else:
        # No aspirated swallows for this participant
        aspirating_numbers = set()

    aspirating_dict[participant] = aspirating_numbers


# List and sort WAV files
wav_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.wav')]
sorted_files = sorted(wav_files, key=sort_key)

features_list = []
labels_list = []


for file in sorted_files:
    match = re.match(r"(\d+)_(\d+)\.wav", file)
    if not match:
        print(f"Filename {file} does not match expected pattern, skipping.")
        continue

    participant_num = int(match.group(1))
    swallow_num = int(match.group(2))
    participant_id = f"P{participant_num}"  # Add 'P' prefix to match CSV participant IDs

    # Assign label: 1 if swallow is aspirated, else 0
    label_set = aspirating_dict.get(participant_id, set())
    label = 1 if swallow_num in label_set else 0

    try:
        fs, audio = wav.read(os.path.join(input_folder, file))
        if audio.ndim > 1:
            audio = audio[:, 0]  # Use first channel if stereo
        if len(audio) == 0:
            print(f"{file} does not have audio")
            continue

        # Extract Spectral Subband Centroid features
        output = ssc(audio, samplerate=fs, winlen=0.025, winstep=0.0125,
                     nfilt=26, nfft=2048, lowfreq=0, highfreq=fs/2)
        agg_features = aggregation_calculations(output)

        print(f"Processed {file} - Participant: {participant_id}, Swallow #: {swallow_num}, Label: {label}")

        features_list.append(agg_features)
        labels_list.append(label)

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Convert to numpy arrays for ML
X = np.array(features_list)
y = np.array(labels_list)

print(f"Extracted features and labels for {len(X)} audio files.")

# Split data into train/test sets (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


# Apply SMOTE to the training data only
# Smote is applied to the training data to handle class imbalance
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f"Before SMOTE, training class distribution: {np.bincount(y_train)}")
print(f"After SMOTE, training class distribution: {np.bincount(y_train_res)}")

# Scale features
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# Train SVM classifier with polynomial kernel and balanced classes
model = SVC(C=1, gamma='auto', kernel='poly', class_weight="balanced")
model.fit(X_train_res, y_train_res)



# Predict and evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

