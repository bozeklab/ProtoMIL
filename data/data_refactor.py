import os
import re
import glob
import shutil
from concurrent.futures import ThreadPoolExecutor

def get_all_tifs(key):
    dirs = glob.glob(key, recursive=True)
    dirs = [d for d in dirs if 'mask' not in d]
    dirs.sort()
    print(f'{len(dirs)=}')
    return dirs

def process_file(file_in, new_dir):
    file_name = new_dir
    if '- - cl1/' in file_in:
        file_name += 'cl1/'
    elif '- - cl19/' in file_in:
        file_name += 'cl19/'
    else:
        file_name += 'fl_fl/'

    if 'E1 fl cl' in file_in:
        file_name += 'e1_'
    elif 'E2 fl cl' in file_in:
        file_name += 'e2_'
    else:
        file_name += 'e3_'

    part = file_in[-12:-8] if '.dm4.' in file_in else file_in[-8:-4]
    num = int(re.findall(r'\d+', part)[0])
    file_name += str(num)

    if not os.path.exists(file_name + '.tif'):
        file_name += '.tif'
        shutil.copyfile(file_in, file_name)
    else:
        i = 1
        while os.path.exists(file_name + '_' + str(i) + '.tif'):
            i += 1
        file_name = file_name + '_' + str(i) + '.tif'
        shutil.copyfile(file_in, file_name)
    print(file_name)

if __name__ == '__main__':
    new_dir = 'mito_scale/'
    old_dir = 'EM_pictures'
    dirs = ['', 'cl1', 'cl19', 'fl_fl']
    for d in dirs:
        os.makedirs(new_dir + d, exist_ok=True)

    files = get_all_tifs(old_dir + '/**/10kX magnification/**/*.tif')
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file_in, new_dir) for file_in in files]
        for future in futures:
            future.result()
