import scipy.io.wavfile as wav
import h5py
import os
import numpy as np

# Folder containing the WAV audio files
input_folder = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/mono_audio1"

# Folder to save extracted swallowing segments
output_folder = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/swallowing_segments_paed"

# Path to the HDF5 file containing swallow labels
h5_file = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/Scripts/labels.h5"

# Folder to save extracted non-swallowing segments
non_swallow_output_folder = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/non_swallowing_segments_paed"


def extract_digits(s):
    """Extract digits from a string to match HDF5 group names."""
    return ''.join(filter(str.isdigit, s))


# Open the HDF5 label file for reading
with h5py.File(h5_file, "r") as f:

    # Iterate over all WAV files in the input folder
    for file in os.listdir(input_folder):
        if not file.lower().endswith('.wav'):
            continue  # skip non-wav files

        # Extract participant ID digits only to match HDF5 groups
        participant_str = file.split('_')[0]  # e.g., "MBS 64"
        participant_id = extract_digits(participant_str)  # e.g., "64"

        start_samples_path = f"labels/{participant_id}/start/samples"
        end_samples_path = f"labels/{participant_id}/stop/samples"

        try:
            # Read swallow start and stop sample indices from HDF5
            start_samples = f[start_samples_path][:]
            end_samples = f[end_samples_path][:]

            # Read the audio file
            fs, audio = wav.read(os.path.join(input_folder, file))

            # Extract and save each swallowing segment
            for i in range(len(start_samples)):
                swallowing = audio[start_samples[i]:end_samples[i]]
                new_file = f"{participant_id}_{i + 1}.wav"
                output_path = os.path.join(output_folder, new_file)
                wav.write(output_path, fs, swallowing)

            # Extract non-swallowing segments by collecting the gaps between swallowing segments
            non_swallow_segments = []
            prev_end = 0

            for i in range(len(start_samples)):
                if start_samples[i] > prev_end:
                    non_swallow_segments.append(audio[prev_end:start_samples[i]])
                prev_end = end_samples[i]

            # Append audio after the last swallowing segment if any
            if prev_end < len(audio):
                non_swallow_segments.append(audio[prev_end:])

            # Concatenate and save non-swallowing audio if any segments exist
            if non_swallow_segments:
                non_swallow_audio = np.concatenate(non_swallow_segments)
                non_swallow_file = f"{participant_id}_non_swallow.wav"
                non_swallow_path = os.path.join(non_swallow_output_folder, non_swallow_file)
                wav.write(non_swallow_path, fs, non_swallow_audio)

        except KeyError:
            print(f"Labels missing for participant {participant_id}, skipping.")
        except Exception as e:
            print(f"Error processing participant {participant_id}: {e}")
