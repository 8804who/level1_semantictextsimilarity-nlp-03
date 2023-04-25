import pandas as pd

df = pd.read_csv('./data/train.csv')
print(len(df['sentence_1']))

bins = list(range(0,130,10))
bins_label = [str(x)+"~"+str(x+9)for x in bins]
length1=pd.cut(df['sentence_1'].str.len(), bins, right=False, labels=bins_label[:-1])
length2=pd.cut(df['sentence_2'].str.len(), bins, right=False, labels=bins_label[:-1])
print(length1.value_counts(sort=False))
print(length2.value_counts(sort=False))