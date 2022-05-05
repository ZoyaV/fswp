
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
convlstm_path = osp.join(this_dir, 'convlstm')
unet_path = osp.join(this_dir, 'unet')
add_path(convlstm_path)
add_path(unet_path)
