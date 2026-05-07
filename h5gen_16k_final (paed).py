import numpy as np
import h5py as h5
import scipy.io.wavfile as wf
import pandas as pd
import os

rawSave = 1  # Flag to control whether to save raw audio data again (1 = save, 0 = skip)

def convertTimestamp(timestamp, Fs):
    """
    Convert a timestamp string or number to time in seconds and corresponding sample index.
    Supports multiple timestamp formats:
    - Numeric (float or int)
    - "mm:ss.sss"
    - "mm.ss.ms" (with two dots)
    Also fixes common typos like ';' instead of ':'.
    
    Returns:
        time_in_seconds (float): time value in seconds
        sample_index (int): corresponding sample index in audio (based on sampling freq Fs)
    """
    if isinstance(timestamp, float) or isinstance(timestamp, int):
        # Timestamp is already a number
        time_in_seconds = float(timestamp)
    else:
        # Clean string: remove extra quotes and whitespace
        timestamp = str(timestamp).strip().strip("'").strip('"')

        # Fix common typo: replace ';' with ':'
        timestamp = timestamp.replace(';', ':')

        try:
            if '.' in timestamp and timestamp.count('.') == 2 and ':' not in timestamp:
                # Format mm.ss.ms (e.g. "01.23.456")
                parts = timestamp.split('.')
                if len(parts) == 3:
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    milliseconds = float(parts[2])
                    time_in_seconds = minutes * 60 + seconds + milliseconds / 1000
                else:
                    raise ValueError
            elif ':' in timestamp:
                # Format mm:ss.sss (e.g. "01:23.456")
                parts = timestamp.split(':')
                if len(parts) == 2:
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    time_in_seconds = minutes * 60 + seconds
                else:
                    raise ValueError
            else:
                # Just seconds as a string number
                time_in_seconds = float(timestamp)
        except ValueError:
            # Unexpected format: warn and default to 0 seconds
            print(f"Warning: Unexpected timestamp format '{timestamp}'. Treating as 0 seconds.")
            time_in_seconds = 0.0

    # Convert time in seconds to sample index based on sampling frequency Fs
    sample_index = int(round(time_in_seconds * Fs))

    return time_in_seconds, sample_index


if __name__ == "__main__":
    # Paths to data and labels
    dataPath = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/mono_audio_vfss"

    labelPath = "/Users/samtruong/Library/CloudStorage/OneDrive-GriffithUniversity/Desktop/3rd year/Stephen's Research/paed_vfss/vinod_edited_spreadsheet-VFSSdatset-22Aug2022.csv"

    # Read CSV label file into pandas DataFrame
    labelcsv = pd.read_csv(labelPath)

    # Clean column names: remove whitespace and quotes
    labelcsv.columns = [col.strip().strip("'").strip('"') for col in labelcsv.columns]
    print([f"'{col}'" for col in labelcsv.columns.tolist()])  # Print columns for verification

    # Open an HDF5 file to save raw audio and labels
    with h5.File("labels.h5", "w") as v5:
        # Iterate over each participant's row in the CSV
        for idx, row in labelcsv.iterrows():
            p = ''.join(filter(str.isdigit, str(row['Participant'])))  # extracts digits only, e.g. "9"
            filename = f"MBS {p}_mono.wav"  # Expected audio filename
            filepath = os.path.join(dataPath, filename)

            print("Reading ", filename)  # Log current file being processed

            try:
                # Read audio file: Fs = sampling freq, x = audio samples
                (Fs, x) = wf.read(filepath)
            except (FileNotFoundError, OSError) as e:
                # Warn and skip if file not found or unreadable
                print(f"Warning: Audio file not found or unreadable: {filepath}. Skipping participant {p}.")
                continue

            Nx = len(x)  # Total number of samples in audio
            print("Number of samples: ", Nx)

            if rawSave == 1:
                # Save raw audio data into HDF5 dataset under raw/{participant}_1
                dset = v5.create_dataset(f"raw/{p}_1", (Nx,), dtype='int16', fletcher32=True)
                if x.ndim > 1:
                    # Stereo audio: save only left channel
                    dset[...] = x[:, 0]
                else:
                    # Mono audio: save as is
                    dset[...] = x

            numSwallows = row['Total swallows']  # Number of swallows annotated
            print("\tNumber of swallows = ", numSwallows)

            # Initialize arrays to hold start/stop times and sample indices for swallows
            startTime = np.zeros(int(numSwallows), dtype=float)
            stopTime = np.zeros(int(numSwallows), dtype=float)
            startSample = np.zeros(int(numSwallows), dtype=int)
            stopSample = np.zeros(int(numSwallows), dtype=int)

            # Initialize class label array for entire audio (0 = no swallow, 1 = swallow)
            classLabels = np.zeros(Nx)

            # Process each swallow event for this participant
            for swallow in range(int(numSwallows)):
                column1 = f"Swallow Number {swallow + 1} Start"  # CSV column for start time
                column2 = f"Swallow Number {swallow + 1} Stop"   # CSV column for stop time

                startStamp = row[column1]  # Get start timestamp string/value
                stopStamp = row[column2]   # Get stop timestamp string/value

                # Convert timestamps to seconds and sample indices
                (startTime[swallow], startSample[swallow]) = convertTimestamp(startStamp, Fs)
                (stopTime[swallow], stopSample[swallow]) = convertTimestamp(stopStamp, Fs)

                # Mark samples between start and stop as swallow (1)
                classLabels[startSample[swallow]:stopSample[swallow]] = 1

            # Save swallow times and sample indices into HDF5 datasets under labels/{participant}/...
            v5.create_dataset(f"labels/{p}/start/times", data=startTime, fletcher32=True)
            v5.create_dataset(f"labels/{p}/stop/times", data=stopTime, fletcher32=True)
            v5.create_dataset(f"labels/{p}/start/samples", data=startSample, fletcher32=True)
            v5.create_dataset(f"labels/{p}/stop/samples", data=stopSample, fletcher32=True)

            # Save the binary class label array (swallow presence over time)
            v5.create_dataset(f"labels/{p}/class", data=classLabels, dtype="int8", fletcher32=True)

