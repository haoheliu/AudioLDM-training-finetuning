import os
from tqdm import tqdm
import json

# Define the expected structure as a dictionary
expected_structure = {
    "checkpoints": [
        "audiomae_16k_128bins.ckpt",
        "clap_music_speech_audioset_epoch_15_esc_89.98.pt",
        "hifigan_16k_64bins.ckpt",
        "hifigan_16k_64bins.json",
        "hifigan_48k_256bins.ckpt",
        "hifigan_48k_256bins.json",
        "vae_mel_16k_64bins.ckpt"
    ],
    "dataset": {
        "audioset": [
            "zip_audios"
        ],
        "metadata": [
            "audiocaps",
            "dataset_root.json"
        ]
    }
}

def load_json(fname):
    with open(fname,'r') as f:
        data = json.load(f)
        return data

def validate_structure(root_path, expected):
    for name, content in expected.items():
        path = os.path.join(root_path, name)
        if isinstance(content, list):
            # Check files
            for file_name in content:
                file_path = os.path.join(path, file_name)
                if not os.path.exists(file_path):
                    return False, f"Missing file: {file_path}"
        elif isinstance(content, dict):
            # Check subdirectories
            if not os.path.isdir(path):
                return False, f"Missing directory: {path}"
            # Check the rest of the content which should be subdirectories
            valid, message = validate_structure(path, content)
            if not valid:
                return False, message
        else:
            return False, "Invalid structure definition"
    return True, "All files and directories are present"

copied_folder_path = 'data' 
valid, message = validate_structure(copied_folder_path, expected_structure)
print(message)
print("Checking the validity of the audio datasets")
metadata_json_list = ["data/dataset/metadata/audiocaps/datafiles/audiocaps_train_label.json", 
                        "data/dataset/metadata/audiocaps/testset_subset/audiocaps_test_nonrepeat_subset_0.json"]
missing_files_count = 0
for metadata_json in metadata_json_list:
    metadata = load_json(metadata_json)["data"]
    for datapoint in tqdm(metadata):
        audiopath = os.path.join("./data/dataset/audioset", datapoint["wav"])
        if not os.path.exists(audiopath):
            print(f"Missing file: {audiopath}")
            missing_files_count += 1

if missing_files_count == 0:
    print("All audio files are present. You are good to go!")
else:
    print(f"{missing_files_count} audio files are missing")
