import pandas as pd
df = pd.read_csv('./data/train.csv')

df['sentence_1'] = df['sentence_1'].str.replace(pat=r'[^ㄱ-ㅣ가-힣 ]+', repl= r'', regex=True)
df['sentence_2'] = df['sentence_2'].str.replace(pat=r'[^ㄱ-ㅣ가-힣 ]+', repl= r'', regex=True)

df.to_csv('./data/onlyKor.csv')