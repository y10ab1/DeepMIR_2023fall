# Assume that your command is `demucs --mp3 --two-stems vocals -n mdx_extra "track with space.mp3"`
# The following codes are same as the command above:
import os
import glob
import demucs.separate


dirpath = "artist20"

# read train and valid data path from txt
train_path = []
valid_path = []
with open("data_path/train.txt", "r") as f:
    for line in f:
        train_path.append(line.strip().split(",")[0])

with open("data_path/validation.txt", "r") as f:
    for line in f:
        valid_path.append(line.strip().split(",")[0])
        
# output dir for seperated vocals and accompaniment
train_separated = "artist20/train_separated"
valid_separated = "artist20/valid_separated"
test_separated = "artist20/test_separated"

failed_list = []

        

# seperate vocals and accompaniment for train data
for path in train_path:
    
    # DeepMIR_2023fall/hw1/artist20/train_separated/queen/A_Night_At_the_Opera/mdx_extra/01-Death_On_Two_Legs_Dedicated_To_
    # if the path has been seperated, skip it
    if os.path.exists(os.path.join(train_separated, path.split("/")[-3], path.split("/")[-2], 'mdx_extra', path.split("/")[-1].split(".")[0])):
        print("skip", path)
        continue
    
    # get artist name
    artist = path.split("/")[-3]
    # get album name
    album = path.split("/")[-2]
    # get song name
    song = path.split("/")[-1].split(".")[0]
    # get output path
    vocal_output = os.path.join(train_separated, artist, album)
    # create output dir
    os.makedirs(vocal_output, exist_ok=True)
    # get input path
    input_path = os.path.join(dirpath, path)
    print(input_path, vocal_output)
    
    
    
    # seperate vocals and accompaniment
    try:
        demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", input_path, "--out", vocal_output])
    except:
        print("error", input_path)
        failed_list.append(input_path)
    
    # free up memory
    del artist, album, song, vocal_output, input_path
    
# seperate vocals and accompaniment for valid data
for path in valid_path:
    
    # if the path has been seperated, skip it
    if os.path.exists(os.path.join(valid_separated, path.split("/")[-3], path.split("/")[-2], 'mdx_extra', path.split("/")[-1].split(".")[0])):
        print("skip", path)
        continue
    
    # get artist name
    artist = path.split("/")[-3]
    # get album name
    album = path.split("/")[-2]
    # get song name
    song = path.split("/")[-1].split(".")[0]
    # get output path
    vocal_output = os.path.join(valid_separated, artist, album)
    # create output dir
    os.makedirs(vocal_output, exist_ok=True)
    # get input path
    input_path = os.path.join(dirpath, path)
    print(input_path, vocal_output)
    
    # seperate vocals and accompaniment
    try:
        demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", input_path, "--out", vocal_output])
    except:
        print("error", input_path)
        failed_list.append(input_path)
        
    # free up memory
    del artist, album, song, vocal_output, input_path
    
# seperate vocals and accompaniment for test data (mp3 or wav)
for path in glob.glob("artist20/test/*"):
    
    # if the path has been seperated, skip it
    if os.path.exists(os.path.join(test_separated, path.split("/")[-1].split(".")[0], "mdx_extra")):
        print("skip", path)
        continue
    
    idx = path.split("/")[-1].split(".")[0]
    # get output path
    vocal_output = os.path.join(test_separated, idx)
    # create output dir
    os.makedirs(vocal_output, exist_ok=True)
    # get input path
    input_path = path
    print(input_path,vocal_output)
    # seperate vocals and accompaniment
    try:
        demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", input_path, "--out", vocal_output])
    except:
        print("error", input_path)
        failed_list.append(input_path)
    
    # free up memory
    del idx, vocal_output, input_path
    
# save failed list
with open("failed_list.txt", "w") as f:
    for path in failed_list:
        f.write(path+"\n")
        
        