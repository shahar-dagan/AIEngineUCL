import json
import shutil
import os

def remove_model_cache():
    save_dir = "./model_cache"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print("Folder ./model_cache removed successfully.")
    else:
        print("Folder ./model_cache does not exist.")

def copy_folder(src, dst):
    """
    Copies an entire folder and its contents to a new location.
    
    :param src: Path to the source folder
    :param dst: Path to the destination folder with a new name
    """
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source folder '{src}' does not exist.")
    
    shutil.copytree(src, dst)
    print(f"Copied '{src}' to '{dst}'")

def push_model(message):



    with open("./model_cache/model_metadata.json", "r") as file:
        file_data = json.load(file)

    time_stamp = file_data["timestamp"]
    file_data["message"] = message


    with open("./model_cache/model_metadata.json", "w") as file:
        json.dump(file_data, file)


    copy_folder(
        "./model_cache", 
        f"./../server/models/model_{time_stamp}"
    )

    # with open("")
    

if __name__ == "__main__":
    push_model()