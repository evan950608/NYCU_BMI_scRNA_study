import configparser
import os

def get_config():
    """
    Loads the config.ini file and returns the ConfigParser object.
    """
    # Get the path of the current script (this will work regardless of where the script is called from)
    current_dir = os.path.dirname(os.path.abspath(__file__))  # directory of py file
    # print('Current dir:', current_dir)

    # Assume config.ini is located in the evan_home root directory
    root_dir = os.path.abspath(os.path.join(current_dir, "../.."))  # evan_home directory
    # print('Root dir:', root_dir)
    config_path = os.path.join(root_dir, "config.ini")

    # Load the config file
    config = configparser.ConfigParser()
    config.read(config_path)

    return config

def get_dataset_path():
    """
    Returns the full path of a dataset file based on the config.ini paths.
    
    Parameters:
    - dataset_name (str): Name of the dataset file.

    Returns:
    - str: Full path to the dataset.
    """
    config = get_config()
    dataset_dir = config.get("Paths", "dataset_dir")

    # Construct full path using the dataset directory from config.ini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, "../.."))  # evan_home directory
    return os.path.join(root_dir, dataset_dir)

def get_source_code_path():
    config = get_config()
    source_code_dir = config.get("Paths", "source_code_dir")

    # Construct full path using the dataset directory from config.ini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, "../.."))  # evan_home directory
    return os.path.join(root_dir , source_code_dir)

### Test
# if __name__ == "__main__":
#     config = get_config()
#     print(type(config))
#     print('=====')
#     data_dir = get_dataset_path()
#     print('Dataset path: ', data_dir)
#     print('=====')
#     source_code_dir = get_source_code_path()
#     print('Source code path: ', source_code_dir)