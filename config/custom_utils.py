import os
from pathlib import Path


def get_cuda_device_environ():
    try:
        cuda_env = os.environ['CUDA_VISIBLE_DEVICES']
        return str(cuda_env)
    except KeyError as e:
        return 'NA'


def get_config_directory():
    # Get the path to the script's directory
    script_dir = Path(__file__).resolve().parent

    # Navigate up to the MyProject directory
    project_dir = script_dir.parent

    # Define the config directory path
    config_dir = os.path.join(project_dir, 'config')

    # Convert to string if needed
    config_dir_path = str(config_dir)
    return config_dir_path
