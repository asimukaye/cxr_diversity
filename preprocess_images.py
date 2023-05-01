import cv2
from multiprocessing import pool
from multiprocessing.dummy import Pool as ThreadPool
import pandas as pd
from tqdm import tqdm 
import os


root = '/home/asim.ukaye/physionet.org/files/mimic-cxr-jpg/2.0.0/'

df_data = pd.read_csv(os.path.join(root, 'mimic-cxr-2.0.0-dataloader.csv'), header=0, sep=',')

df_path = 'files/p' \
        + df_data['subject_id'].floordiv(1000000).astype(str)\
        + '/p' + df_data['subject_id'].astype(str)\
        + '/s'+ df_data['study_id'].astype(str)\
        + '/' + df_data['dicom_id'] +'.jpg'

img_files = df_path.to_list()



def resize_and_save(filename):
    root_in = '/home/asim.ukaye/physionet.org/files/mimic-cxr-jpg/2.0.0/'
    root_out = '/home/asim.ukaye/ml_proj/mimic_cxr_pa_resized/'
    dir_path = os.path.dirname(root_out + filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    in_path = root_in + filename
    out_path = root_out + filename

    im = cv2.imread(in_path)
    small_im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
    
    cv2.imwrite(out_path, small_im)

pool = ThreadPool(24)
pool.map(resize_and_save, img_files)

# for i in range(10):
#     resize_and_save(img_files[i])