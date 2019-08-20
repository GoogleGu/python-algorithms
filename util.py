import os
from config import DATA_ROOT


def get_data_file_path(data_file_name):
    return DATA_ROOT + os.sep + data_file_name


