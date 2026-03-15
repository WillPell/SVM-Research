import scipy.io.wavfile as wav
import h5py
import os
import numpy as np

# Folder containing the WAV audio files
input_folder = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/mono_audio(cfe)"

# Path to the HDF5 file containing swallow labels
h5_file = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/Scripts/labels(cfe).h5"

# Folder to save extracted swallowing segments
output_folder = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/swallowing_segments_cfe"

# Folder to save extracted non-swallowing segments
non_swallow_output_folder = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/non_swallowing_segments_cfe"


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

        # include the P here if doing for cfe, no P otherwise
        start_samples_path = f"labels/P{participant_id}/start/samples"
        end_samples_path = f"labels/P{participant_id}/stop/samples"

        try:
            # Read swallow start and stop sample indices from HDF5
            start_samples = f[start_samples_path][:]
            end_samples = f[end_samples_path][:]

            # Read the audio file
            fs, audio = wav.read(os.path.join(input_folder, file))

            # Convert sample indices to time in seconds
            start_times_sec = start_samples / fs
            end_times_sec = end_samples / fs

            # Calculate gaps between consecutive swallows: start of next swallow minus end of current swallow
            gap_times = start_times_sec[1:] - end_times_sec[:-1]

            print(f"Participant {participant_id} swallow gap times (seconds):")
            for i, gap in enumerate(gap_times, start=1):
                print(f"  Gap between swallow {i} end and swallow {i+1} start: {gap:.3f} seconds")

            # Adjust swallowing segments by including part of the gap after each swallow (except last)
            adjusted_end_samples = np.copy(end_samples)  # copy to modify

            for i in range(len(gap_times)):
                gap = gap_times[i]
                gap_samples = int(round(gap * fs))

                if gap > 0.5:
                    # Add 0.5 seconds worth of samples to the end of the swallow segment
                    add_samples = int(round(0.5 * fs))
                else:
                    # Add half the gap duration in samples 
                    add_samples = gap_samples // 2

                # Calculate new end sample, ensuring it doesn't overlap next swallow start or exceed audio length
                proposed_end = end_samples[i] + add_samples

                # Limit proposed_end to start of next swallow to avoid overlap
                max_end = start_samples[i + 1] if i + 1 < len(start_samples) else len(audio)

                if proposed_end > max_end:
                    proposed_end = max_end

                # Also ensure proposed_end does not exceed audio length
                if proposed_end > len(audio):
                    proposed_end = len(audio)

                adjusted_end_samples[i] = proposed_end

            # For the last swallow, no gap after it; keep original end
            # adjusted_end_samples[-1] already equals end_samples[-1]

            # Extract and save each adjusted swallowing segment
            for i in range(len(start_samples)):
                swallowing = audio[start_samples[i]:adjusted_end_samples[i]]
                new_file = f"{participant_id}_{i + 1}.wav"
                output_path = os.path.join(output_folder, new_file)
                wav.write(output_path, fs, swallowing)

            # Extract non-swallowing segments by collecting the gaps between adjusted swallowing segments
            non_swallow_segments = []
            prev_end = 0

            for i in range(len(start_samples)):
                if start_samples[i] > prev_end:
                    non_swallow_segments.append(audio[prev_end:start_samples[i]])
                prev_end = adjusted_end_samples[i]

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
