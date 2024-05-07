import os
import sys
import re
import glob
import shutil

# WARNING before using download mitochondria dataset from sciebo and name it mito_org

def get_all_tifs(path):
    dirs = glob.glob(path + '/**/*.tif', recursive=True)
    dirs = [d for d in dirs if 'mask' not in d]
    dirs.sort()
    print(len(dirs))
    return dirs

if __name__ == '__main__':
    new_dir = 'mito/'
    old_dir = 'mito_org'
    dirs = ['', 'cl1', 'cl19', 'fl_fl']
    for d in dirs:
        os.makedirs(new_dir + d, exist_ok=True)

    for file_in in get_all_tifs(old_dir):
        file_name = new_dir
        if '/cl1/' in file_in:
            file_name += 'cl1/'
        elif '/cl19/' in file_in:
            file_name += 'cl19/'
        else:
            file_name += 'fl_fl/'

        if '/E1/' in file_in:
            file_name += 'e1_'
        elif '/E2/' in file_in:
            file_name += 'e2_'
        else:
            file_name += 'e3_'

        part = file_in[-12:-8] if '.dm4.' in file_in else file_in[-8:-4]
        num = int(re.findall(r'\d+', part)[0])

        file_name += str(num) + '.tif'
        shutil.copyfile(file_in, file_name)
        print(file_name)
