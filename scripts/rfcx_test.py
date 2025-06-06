import hashlib
import os
import pathlib
import subprocess
import sys

from plumbum import FG, local

# Error while downloading flac files...
# kaggle competitions download rfcx-species-audio-detection -f train/032cb2915.flac
# flac -t 032cb2915.flac   # Returns error

# If I try to fix the file
# flac -F -t 032cb2915.flac  --> Gets a lot of errors and "ERROR, MD5 signature mismatch"

# I can fix the file, but then the checksum is different...
# flac --verify --decode-through-errors --preserve-modtime -o fixed.flac 032cb2915.flac

# I also tried to convert it with ffmpeg instead: no errors (even though flac -t *.flac) returns an error, but a different checksum (it does too with a healthy file), probably because sox and ffmpeg have slight differences

dataset = {}

def get_md5(file_path:pathlib.Path) -> str:
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()

file_names = ["032cb2915.flac", "0a4e7e350.flac"]

# target_hash_value = ["cfed055fc2d78478124efa9dae38852f", "7d1663f53d4cd54d01e63caa7235b188"]
local["mkdir"]["-p", "data/rfcx/wav"]()
dest_dir = pathlib.Path("data/rfcx/wav")

for file_name in file_names:
    if pathlib.Path(f"data/rfcx/{file_name}").exists():
        pathlib.Path(f"data/rfcx/{file_name}").unlink()
    if pathlib.Path(f"data/rfcx/{file_name}.zip").exists():
        pathlib.Path(f"data/rfcx/{file_name}.zip").unlink()
    
    (
        local["kaggle"][
            "competitions", "download", "-p", "data/rfcx", "rfcx-species-audio-detection", "-f", f"train/{file_name}"
        ]
        & FG
    )
    local["unzip"][f"data/rfcx/{file_name}.zip", "-d", "data/rfcx/"] & FG


for i, file_name in enumerate(file_names):
    src_file = pathlib.Path(f"/home/cangonin/Documents/github/beans/data/rfcx/{file_name}")
    dest_file = dest_dir / (src_file.stem + ".wav")
    if os.path.exists(dest_file):
        dest_file.unlink()
    if not os.path.exists(dest_file):
        print(f"Converting {src_file} ...", file=sys.stderr)
        subprocess.run(["sox", src_file, "-r 48000", "-R", dest_file])
    dataset[src_file.stem] = {
        "path": str(dest_file),
        "length": 9999,
        "annotations": [],
    }
    # print(f"File {file_name}: \n MD5 hash of created wav file: {get_md5(dest_file)} \n MD5 hash in the hash file: {target_hash_value[i]}")
