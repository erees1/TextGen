# -*- coding: utf-8 -*-
import shutil
def copy_to_local(src_path, dest_path):
    shutil.copytree(src_path, dest_path)
