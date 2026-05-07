import scipy.io.wavfile as wav
from python_speech_features import ssc
import os
import numpy as np
import pandas as pd
import re
from collections import defaultdict


# Import SMOTE from imbalanced-learn
from imblearn.over_sampling import SMOTE


# SVM model libraries (for the model stats and splitting of the data)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


### Helper functions ###

def aggregation_calculations(output):
    # Calculate mean and std deviation for each feature across all frames
    mean = np.mean(output, axis=0)
    std = np.std(output, axis=0)
    return np.concatenate([mean, std])

def sort_key(filename):
    # Sort files by participant and swallow number extracted from filename without 'P' prefix
    file_clean = filename.strip()  # Remove whitespace
    file_clean = os.path.basename(file_clean)  # Just filename
    match = re.match(r"(\d+)_(\d+)\.wav", file_clean, re.IGNORECASE)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    else:
        return (float('inf'), float('inf'))


### TRAINING SET (VFSS) ###

input_folder = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/swallowing_segments_vfss"

labelling_csv = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/paed_vfss/vinod_edited_spreadsheet-VFSSdatset-22Aug2022.csv"


labels_df = pd.read_csv(labelling_csv)
labels_df.columns = labels_df.columns.str.strip()

# Build dictionary participant -> set of aspirated swallow numbers
aspirating_dict = {}
for idx, row in labels_df.iterrows():
    participant = str(row['Participant']).strip()
    aspirating_flag = row.get('Aspirating', 0)
    which_swallows = row.get('Which Swallows', "")

    if aspirating_flag == 1:
        aspirating_numbers = set()
        if pd.notna(which_swallows):
            which_swallows_str = str(which_swallows).strip()
            if which_swallows_str:
                for x in which_swallows_str.split(','):
                    x = x.strip()
                    try:
                        aspirating_numbers.add(int(float(x)))
                    except ValueError:
                        pass
    else:
        aspirating_numbers = set()

    aspirating_dict[participant] = aspirating_numbers

# Process training audio files
wav_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.wav')]
sorted_files = sorted(wav_files, key=sort_key)

features_list = []
labels_list = []


# going through all of the files in the directory
for file in sorted_files:
    file_clean = file.strip()
    match = re.match(r"(\d+)_(\d+)\.wav", file_clean)
    if not match:
        print(f"Filename {file} does not match expected pattern, skipping.")
        continue

    participant_num = int(match.group(1))
    swallow_num = int(match.group(2))
    participant_id = f"P{participant_num}"

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

        features_list.append(agg_features)
        labels_list.append(label)

    except Exception as e:
        print(f"Error processing {file}: {e}")


### TESTING SET (CFE) ###

testing_folder = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/swallowing_segments_cfe"

test_labelling_csv = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/Swallowing1.csv"

test_labels_df = pd.read_csv(test_labelling_csv)
test_labels_df.columns = test_labels_df.columns.str.strip()

# Build dictionary participant -> set of aspirated swallow numbers for test set
test_aspirating_dict = {}
for idx, row in test_labels_df.iterrows():
    participant = str(row['Participant']).strip()
    aspirating_str = row.get('Aspirating', "")

    aspirating_numbers = set()
    if pd.notna(aspirating_str) and aspirating_str.strip() != "":
        for x in str(aspirating_str).split(','):
            x = x.strip()
            try:
                aspirating_numbers.add(int(float(x)))
            except ValueError:
                pass

    test_aspirating_dict[participant] = aspirating_numbers

# Process testing audio files
test_wav_files = [f for f in os.listdir(testing_folder) if f.lower().endswith('.wav')]
test_sorted_files = sorted(test_wav_files, key=sort_key)

test_features_list = []
test_labels_list = []

for file in test_sorted_files:
    match = re.match(r"(\d+)_(\d+)\.wav", file)
    if not match:
        print(f"Filename {file} does not match expected pattern, skipping.")
        continue

    participant_num = int(match.group(1))
    swallow_num = int(match.group(2))
    participant_id = f"P{participant_num}"

    label_set = test_aspirating_dict.get(participant_id, set())
    label = 1 if swallow_num in label_set else 0

    try:
        fs, audio = wav.read(os.path.join(testing_folder, file))
        if audio.ndim > 1:
            audio = audio[:, 0]
        if len(audio) == 0:
            print(f"{file} does not have audio")
            continue

        output = ssc(audio, samplerate=fs, winlen=0.025, winstep=0.0125,
                     nfilt=26, nfft=2048, lowfreq=0, highfreq=fs/2)
        agg_features = aggregation_calculations(output)

        print(f"Processed {file} - Participant: {participant_id}, Swallow #: {swallow_num}, Label: {label}")

        test_features_list.append(agg_features)
        test_labels_list.append(label)

    except Exception as e:
        print(f"Error processing {file}: {e}")

X_test = np.array(test_features_list)
y_test = np.array(test_labels_list)


### MODEL TRAINING AND EVALUATION ###

# Convert training lists to numpy arrays
X = np.array(features_list)
y = np.array(labels_list)

print(f"Extracted features and labels for {len(X)} training audio files.")
print(f"Extracted features and labels for {len(X_test)} testing audio files.")

# Split training data into train/validation sets (optional)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Apply SMOTE only on training data to balance classes
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


print(f"Before SMOTE, training class distribution: {np.bincount(y_train)}")
print(f"After SMOTE, training class distribution: {np.bincount(y_train_res)}")

# Train SVM classifier with polynomial kernel and balanced classes
model = SVC(C=1, gamma='auto', kernel='poly', class_weight="balanced")
model.fit(X_train_res, y_train_res)

# Evaluate on validation set
y_val_pred = model.predict(X_val)


print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))

# Evaluate on separate test set
y_test_pred = model.predict(X_test)

print("\nFiles predicted as label 1 by the model:")

for file, pred in zip(test_sorted_files, y_test_pred):
    if pred == 1:
        print(file)

print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Test Classification Report:\n", classification_report(y_test, y_test_pred))




# Putting into the excel sheet
# Step 1: Group swallow numbers predicted as 1 by participant
predicted_ones = defaultdict(list)

for file, pred in zip(test_sorted_files, y_test_pred):
    if pred == 1:
        match = re.match(r"(\d+)_(\d+)\.wav", file)
        if match:
            participant = f"P{int(match.group(1))}"
            swallow_num = int(match.group(2))
            predicted_ones[participant].append(swallow_num)

# Step 2: Create a DataFrame with participant and comma-separated predicted swallow numbers
pred_summary = pd.DataFrame({
    'Participant': list(predicted_ones.keys()),
    "Predicted_Swallows": [", ".join(map(str, sorted(nums))) for nums in predicted_ones.values()]
})

# Step 3: Load your Excel sheet
test_df = pd.read_csv(test_labelling_csv)
test_df.columns = test_df.columns.str.strip()

# Step 4: Update the existing "Classifier's Pred" column per participant
# Map participant to predicted swallow list
participant_to_pred = dict(zip(pred_summary['Participant'], pred_summary['Predicted_Swallows']))

# Apply mapping to the "Classifier's Pred" column, overwriting existing values per participant
def update_classifier_pred(row):
    participant = row['Participant']
    return participant_to_pred.get(participant, row.get("Classifier's Pred", ""))

test_df["Classifier's Pred"] = test_df.apply(update_classifier_pred, axis=1)

# Step 5: Save updated Excel
output_excel = "test_with_updated_predictions.xlsx"
test_df.to_excel(output_excel, index=False)

print(f"Updated 'Classifier's Pred' column saved to {output_excel}")

