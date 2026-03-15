import os
import librosa
import soundfile as sf

def convert_to_single_channel_librosa(input_file, output_file):
    audio, sr = librosa.load(input_file, sr=None, mono=True)
    sf.write(output_file, audio, sr)
    print(f"Converted and saved to {output_file}")

def convert_all_audio_in_directory(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    # List all wav files directly in the input directory (no subfolders)
    wav_files = [f for f in os.listdir(input_directory) if f.endswith('.wav')]

    if not wav_files:
        print("No WAV files found in the input directory.")
        return

    for wav_file in wav_files:
        input_file_path = os.path.join(input_directory, wav_file)
        # Create output filename by appending '_mono' before extension to avoid overwriting
        base, ext = os.path.splitext(wav_file)
        output_filename = f"{base}_mono{ext}"
        output_file_path = os.path.join(output_directory, output_filename)

        convert_to_single_channel_librosa(input_file_path, output_file_path)

# Directory for the original audio files (all WAV files directly here)
input_directory = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/paed_vfss"

# Where the mono_audio will be stored
output_directory = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/mono_audio_vfss"

convert_all_audio_in_directory(input_directory, output_directory)



