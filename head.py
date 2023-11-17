import sys
import os
from head_orientation_deploy import *

def check_files_exist(file_paths):
    missing_files = [file for file in file_paths if not os.path.exists(file)]
    return missing_files

def bug_checking(input_video_path):
    # List of required files
    required_files = [
        'head_orientation_model/model_architecture.json',
        'head_orientation_model/model_weights.h5',
        'head_orientation_model/scaler_params.json',
        'shape_predictor_68_face_landmarks.dat',
        'head_orientation_deploy.py'
    ]

    if not os.path.exists(input_video_path):
        print(f"Input video file '{input_video_path}' does not exist.")
        return -1

    missing_files = check_files_exist(required_files)

    if missing_files:
        print("The following required files are missing:")
        for file in missing_files:
            print(file)
        return -1

    return 1


def main(input_video_path):
    result = bug_checking(input_video_path)
    
    if result == 1:
        print("All required files and the input video file exist. Proceed with processing.")
        head_orientation_model(input_video_path)
    else:
        print("The code encountered errors. Please resolve them before proceeding.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("[Error] Usage: module_head.py <input_video_path>")
    else:
        video_filename = sys.argv[1]
        main(video_filename)
