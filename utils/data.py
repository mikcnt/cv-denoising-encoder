import os
from itertools import chain

def find_paths(path):
    path = os.path.join(path, "x")
    directories = [
        os.path.join(path, directory) for directory in os.listdir(path)
    ]
    files = list(
        chain(*
            [
                [os.path.join(directory, path) for path in os.listdir(directory)]
                for directory in directories
            ]
        )
    )
    return files

def t_path(image_path, path):
    dir_name = os.path.basename(os.path.dirname(path))
    ext = path.split('.')[-1]
    return os.path.join(image_path, 't', dir_name + '.' + ext)