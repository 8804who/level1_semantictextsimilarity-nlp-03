import pandas as pd
best = pd.read_csv('./data/output/best.csv')
test = pd.read_csv('./output_ensem.csv')

data = {"best":best['target'],"test":test['target']}
df = pd.DataFrame(data)
print(df.corr(method='pearson'))
