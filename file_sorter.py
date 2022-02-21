import pandas as pd
import os, os.path


N_images = sum(len(files) for _, _, files in os.walk('Images1/images'))

id_dataset = pd.read_csv('dat/Data_Entry_2017_v2020.csv')

pos = id_dataset[id_dataset['Finding Labels'].str.contains('Mass')]

non = id_dataset[~id_dataset['Finding Labels'].str.contains('Mass')]

filt_pos = pd.DataFrame(columns=id_dataset.columns)

print(non.shape,pos.shape, id_dataset.shape)

imgs = next(os.walk('Images1/images'))[2]

ids = list(pos['Image Index'])

print(ids)

