import os
import librosa
import soundfile as sf

# convert from stereo to mono
def convert_to_single_channel_librosa(input_file, output_file):
    audio, sr = librosa.load(input_file, sr=None, mono=True)
    sf.write(output_file, audio, sr)
    print(f"Converted and saved to {output_file}")


# function to help convert all the audio in the dictionary
def convert_all_audio_in_directory(input_directory, output_directory):

    # make sure that the directory exists before processing
    os.makedirs(output_directory, exist_ok=True)

    # going through the file in the directory one by one
    for participant_folder in os.listdir(input_directory):
        participant_path = os.path.join(input_directory, participant_folder)
        
        if os.path.isdir(participant_path):
            # List all wav files in the participant folder
            wav_files = [f for f in os.listdir(participant_path) if f.endswith('.wav')]
            
            if 1 <= len(wav_files) <= 2:
                for i, wav_file in enumerate(wav_files, start=1):
                    input_file_path = os.path.join(participant_path, wav_file)
                    # Add index i if there are multiple files to avoid overwriting output
                    if len(wav_files) == 1:
                        output_filename = f"{participant_folder}_mono.wav"
                    else:
                        output_filename = f"{participant_folder}_mono_{i}.wav"
                    output_file_path = os.path.join(output_directory, output_filename)
                    
                    convert_to_single_channel_librosa(input_file_path, output_file_path)
            else:
                print(f"Warning: Expected 1 or 2 wav files in {participant_folder}, found {len(wav_files)}.")

# Directory for the original audio files
input_directory = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/cfe"

# Where the mono_audio will be stored
output_directory = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/mono_audio(cfe)"

convert_all_audio_in_directory(input_directory, output_directory)






