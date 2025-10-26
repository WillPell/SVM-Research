import scipy.io.wavfile as wav
import h5py
import os
import numpy as np


# folder containing the wav file
input_folder = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/mono_audio"

output_folder = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/swallowing_segments"


# h5 file
h5_file = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/Scripts/labels.h5"


non_swallow_output_folder = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/non_swallowing_segments" #


# opening the h5 file
with h5py.File(h5_file, "r") as f:

    for file in os.listdir(input_folder):
        participant_id = file.split('_')[0]

        # getting the path where it stores the samples
        start_samples_path = f"labels/{participant_id}/start/samples"
        end_samples_path = f"labels/{participant_id}/stop/samples"


        try:
            start_samples = f[start_samples_path][:]
            end_samples = f[end_samples_path][:]

            fs, audio = wav.read(os.path.join(input_folder, file))

            #  getting the swallowing length
            for i in range(len(start_samples)):
                swallowing = audio[start_samples[i]:end_samples[i]]
                new_file = f"{participant_id}_{i + 1}.wav"
                output = os.path.join(output_folder, new_file)
                
                wav.write(output, fs, swallowing)
            
            # removing the swallowing from the wav file
            # probably have to cut the swllowing out, then concatenate the previous audio back in
            # find non-swallowing segments
            non_swallow_segments = []
            prev_end = 0

            # appends the "gaps" left by cutting swallowing segments
            for i in range(len(start_samples)):
                # going through the swallowing sample position
                if start_samples[i] > prev_end:

                    # if higher than the previous end, append to the array
                    non_swallow_segments.append(audio[prev_end: start_samples[i]])
                
                # make the new position the end of the swallowing sample
                prev_end = end_samples[i]
                
            # if the non-swallowing position is not at the end
            if prev_end < len(audio):

                # appending the audio between the final end swallowing sound and the end clip
                non_swallow_segments.append(audio[prev_end: len(audio)])

            
        
            # save non-swallowing segments
            if non_swallow_segments:

                # 
                non_swallow_audio = np.concatenate(non_swallow_segments)

                # print(f"{participant_id}: {non_swallow_audio}")

                non_swallow_file = f"{participant_id}_non_swallow.wav"
                non_swallow_path = os.path.join(non_swallow_output_folder, non_swallow_file)
                wav.write(non_swallow_path, fs, non_swallow_audio)

            
        except:
            print(f"{participant_id} does not have label")


        