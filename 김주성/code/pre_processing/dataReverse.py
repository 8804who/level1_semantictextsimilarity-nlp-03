import pandas as pd
df1 = pd.read_csv('./data/train.csv')
df2 = pd.read_csv('./data/train.csv')

df2['sentence_1'], df2['sentence_2']=df2['sentence_2'], df2['sentence_1']

df_Origin_and_Reverse=pd.concat([df1, df2])

df_Origin_and_Reverse.to_csv('./data/df_Origin_and_Reverse.csv')