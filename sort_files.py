import pandas as pd
import os, os.path
import glob
import shutil

"""ONLY RUN THIS ONCE, OTHERWISE CREATES REDUNDANT IMAGES"""

src = 'Images1/images'
dest1 = 'Images1/pos'
dest2 = 'Images1/non'


N_images = sum(len(files) for _, _, files in os.walk('Images1/images'))

id_dataset = pd.read_csv('dat/Data_Entry_2017_v2020.csv')

all_n = len(list(id_dataset['Image Index']))

pos = id_dataset[id_dataset['Finding Labels'].str.contains('Mass')]

non = id_dataset[~id_dataset['Finding Labels'].str.contains('Mass')]

filt_pos = pd.DataFrame(columns=id_dataset.columns)

imgs = next(os.walk('Images1/images'))[2]

ids = list(pos['Image Index'])

n_ids = list(non['Image Index'])

print(all_n,len(ids),len(n_ids))

for name in ids:
    if name in imgs:
        shutil.copy('Images1/images/'+name,dest1)

for nname in n_ids:
    if nname in imgs:
        shutil.copy('Images1/images/' + nname, dest2)


pos_n = sum(len(files) for _, _, files in os.walk('Images1/pos'))
non_n = sum(len(files) for _, _, files in os.walk('Images1/non'))

print(pos_n,non_n)


