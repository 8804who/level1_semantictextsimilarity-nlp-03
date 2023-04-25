import pandas as pd
import os

PATH = './data'
FILE_NAME ='train.csv'
train_data = pd.read_csv(os.path.join(PATH, FILE_NAME))
train_not_zero = train_data[1 <= train_data['label']]
sentence_1 = train_not_zero['sentence_1']
sentence_2 = train_not_zero['sentence_2']
train_not_zero.sentence_2 = sentence_1
train_not_zero.sentence_1 = sentence_2

train_augmented = pd.concat([train_data, train_not_zero], ignore_index=True)
train_augmented.to_csv(os.path.join(PATH, 'train_augmented.csv'), encoding='utf-8', index=False)
