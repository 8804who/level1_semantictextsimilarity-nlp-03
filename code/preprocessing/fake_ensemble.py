import pandas as pd
import math
from numpy import log as ln

def sigmoid(x):
    return 1/(1+math.exp(-x))


def inverse_sigmoid(x):
    return ln(x/(1-x))


paths=['./data/output/KR-ELECTRA-discriminator_JH.csv', './data/output/output_person.csv', './data/output/ensemble_output.csv', './data/output/output_augment.csv']

predicts=[]

for path in paths:
    predicts.append(pd.read_csv(path))

predictions=[]

for i in range(len(predicts[0])):
    num=0
    for j in range(len(predicts)):
        num+=sigmoid(predicts[j]['target'][i])
    avg = (num)/len(predicts)
    val = round(inverse_sigmoid(avg),1)
    if val>5:
        val=5
    if val<0:
        val=0
    predictions.append(val)

output = pd.read_csv('./data/sample_submission.csv')
output['target'] = predictions
output.to_csv('output_ensem.csv', index=False)