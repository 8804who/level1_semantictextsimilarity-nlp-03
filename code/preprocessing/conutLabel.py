import pandas as pd
df = pd.read_csv('./data/train.csv')
bins = list(range(0,7,1))
bins_label = [str(x)+"점대" for x in bins]
df['cut_label']=pd.cut(df['label'], bins, right=False, labels=bins_label[:-1])
print(df['cut_label'].value_counts())
