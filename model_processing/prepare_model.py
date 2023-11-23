import os

def get_latest_model(model_folder):
    model_list =[file for file in os.listdir(model_folder) if file.startswith("u2net_version_")]
    if not model_list:
        latest_model = 'u2net.pth'
    else: 
        sorted_model_files = sorted(model_list, key=lambda x: int(x[len("u2net_version_"):-len(".pth")]))
        latest_model = sorted_model_files[-1]
    return os.path.join(model_folder,latest_model)


def get_latest_version(model_path):
    model_name = model_path.split("/")[-1].split(".")[0]
    latest_version = model_name[-1]
    return latest_version
