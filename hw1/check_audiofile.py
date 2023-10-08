import os
import torchaudio
import glob
from tqdm import tqdm

# Set the directory path
dir_paths = ["artist20/test_seperated", "artist20/train_seperated", "artist20/valid_seperated"]
failed_list = []
# Loop through all files in the directory
for dir_path in dir_paths:
    for filename in tqdm(glob.glob(os.path.join(dir_path, "**"), recursive=True)):
        # Check if the file is an audio file
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            try:
                # Load the audio file using torchaudio
                waveform, sample_rate = torchaudio.load(filename)
                # Print the sample rate and number of channels
                print(f"{filename}: Sample rate - {sample_rate}, Channels - {waveform.shape[0]}")
            except:
                print("error", filename)
                failed_list.append(filename)
                
# Print and save the list of failed files
if len(failed_list) == 0:
    print("No failed files")
else:
    print(failed_list)
    with open("failed_list.txt", "w") as f:
        for item in failed_list:
            f.write("%s\n" % item)