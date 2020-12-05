
import os
import sys

import tarfile

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from tools.download_tools import download_file_from_google_drive

if __name__ == "__main__":

    # Download the full data from the models
    print ("Downloading the models checkpoints   700 MB")
    file_id = '1VJi6CH-Z0hh5LnS2agrNjO81FJ1jW--T'
    destination_pack = 'pretrained_models.tar.gz'

    download_file_from_google_drive(file_id, destination_pack)
    destination_final = '_logs/'
    if not os.path.exists(destination_final):
        os.makedirs(destination_final)

    tf = tarfile.open("pretrained_models.tar.gz")
    tf.extractall(destination_final)
    # Remove the original after moving.
    os.remove("pretrained_models.tar.gz")