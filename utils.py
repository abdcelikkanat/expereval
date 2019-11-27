import os
import sys


def get_file_extension(file_path):
    ext =  os.path.splitext(file_path)[1][1:].strip().lower()
    return ext

