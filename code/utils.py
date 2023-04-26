import re

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def preprop_sent(sentence_lst):
    sentence_lst = [re.sub('!!+', '!!', sentence) for sentence in sentence_lst]
    sentence_lst = [re.sub('\?+' ,'?', sentence) for sentence in sentence_lst]
    sentence_lst = [re.sub('~+', '~', sentence) for sentence in sentence_lst]
    sentence_lst = [re.sub('\.\.+', '...', sentence) for sentence in sentence_lst]
    sentence_lst = [re.sub('ㅎㅎ+', 'ㅎㅎ', sentence) for sentence in sentence_lst]
    sentence_lst = [re.sub('ㅋㅋ+', 'ㅋㅋㅋ', sentence) for sentence in sentence_lst]
    sentence_lst = [re.sub('ㄷㄷ+', 'ㄷㄷ', sentence) for sentence in sentence_lst]
    sentence_lst = [re.sub('…', '...', sentence) for sentence in sentence_lst]
    return sentence_lst

def early_stop(monitor="val_pearson", patience=5, mode="max"):
    early_stop_callback = EarlyStopping(monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode)
    return early_stop_callback


def best_save(top_k=3, monitor="val_pearson", mode="max", model_name='snunlp/KR-ELECTRA-discriminator'):
    checkpoint_callback = ModelCheckpoint(
        dirpath="./model/{model_name}/".format(model_name=model_name.replace('/','_')),
        filename=model_name.replace('/','_') + "-{epoch}-{val_pearson}",
        save_top_k=top_k,
        monitor=monitor,
        mode=mode,
        )
    return checkpoint_callback