import argparse
import time
import pandas as pd

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold
from torch.optim.lr_scheduler import ExponentialLR, StepLR, MultiStepLR
import wandb

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)
    
class KfoldDataloader(pl.LightningDataModule):
    def __init__(self, 
                 model_name, 
                 batch_size, 
                 shuffle, 
                 total_path, test_path, predict_path,
                 k: int = 1,  # fold number
                 split_seed: int = 12345,  # split needs to be always the same for correct cross validation
                 num_splits: int = 20):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.k = k
        self.split_seed = split_seed
        self.num_splits = num_splits

        self.total_path = total_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=100)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        self.bitarget_columns = ['binary-label']

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])

        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
            bitargets = data[self.bitarget_columns].values.tolist()
        except:
            targets = []
            bitargets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets, bitargets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 데이터 준비
            total_data = pd.read_csv(self.total_path)
            total_input, total_targets, binary_targets = self.preprocessing(total_data)
            total_dataset = Dataset(total_input, total_targets)

            # # 데이터셋 num_splits 번 kfold
            # kf = KFold(n_splits=self.num_splits, shuffle=self.shuffle, random_state=self.split_seed)
            # all_splits = [k for k in kf.split(total_dataset)]
            # # k번째 fold 된 데이터셋의 index 선택
            # train_indexes, val_indexes = all_splits[self.k]
            # train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            # statratified kfold
            kf = StratifiedKFold(n_splits=self.num_splits, shuffle=self.shuffle, random_state=self.split_seed)
            all_splits = [k for k in kf.split(total_dataset, binary_targets)]
            # k번째 fold 된 데이터셋의 index 선택
            train_indexes, val_indexes = all_splits[self.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            # fold한 index에 따라 데이터셋 분할
            self.train_dataset = [total_dataset[x] for x in train_indexes] 
            self.val_dataset = [total_dataset[x] for x in val_indexes]

        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets, _ = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets, _ = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
    
class Model(pl.LightningModule):
    def __init__(self, cfg): #model_name, lr):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model_name = cfg['model_name']
        self.lr = cfg['learning_rate']

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name, num_labels=1, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
        # Loss 계산을 위해 사용될 Loss를 호출합니다.
        self.loss_func = torch.nn.MSELoss()# L1Loss()
        

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9,0.999), eps=1e-6)
        # scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.cfg['data_size']//self.cfg['batch_size']*0.1, num_training_steps=self.cfg['data_size']//self.cfg['batch_size'])# * self.cfg['epochs'])
        # scheduler = ExponentialLR(optimizer, gamma=0.95)
        return [optimizer]#, [scheduler]

def train_kfold(args):
    wandb_logger = WandbLogger(project=args.project_name)
    cfg = {
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "epochs": args.max_epoch,
        "shuffle": args.shuffle,
        "learning_rate": args.learning_rate,
        "data_size" : 9874,
        "hidden_dropout": "0.1",
        "attention_dropout":"0.1",
        "ADAMW":"bias=(0.9,0.999),eps=1e-6",
        "loss function":"MSE(L2)",
    }
    wandb.config.update(cfg)
    
    model = Model(cfg)
    
    checkpoint_callback = [
        EarlyStopping(monitor="val_pearson", min_delta=0.00, patience=5, verbose=False, mode="max",),
        ModelCheckpoint(
                        dirpath="./model/{model_name}/".format(model_name=args.model_name.replace('/','_')),
                        filename=args.model_name.replace('/','_') + "-{epoch}-{val_pearson}",
                        monitor="val_pearson",
                        mode="max",
                        save_top_k=3,
                        )
    ]
    results = []
    for k in range(20):
        dataloader = KfoldDataloader(args.model_name, args.batch_size, args.shuffle,
                                     args.total_path, args.test_path, args.predict_path, k=k)

        trainer = pl.Trainer(accelerator='gpu', max_epochs=args.max_epoch, logger=wandb_logger, log_every_n_steps=100, callbacks=checkpoint_callback)
        trainer.fit(model=model, datamodule=dataloader)
        score = trainer.test(model=model, datamodule=dataloader)

        results.extend(score)

    # # dataloader와 model을 생성합니다.
    # dataloader = KfoldDataloader(args.model_name, args.batch_size, args.shuffle, 
    #                              args.total_path, args.test_path, args.predict_path)
    # model = Model(cfg)

    # checkpoint_callback = [
    #     EarlyStopping(monitor="val_pearson", min_delta=0.00, patience=5, verbose=False, mode="max",),
    #     ModelCheckpoint(
    #                     dirpath="./model/{model_name}/".format(model_name=args.model_name.replace('/','_')),
    #                     filename=args.model_name.replace('/','_') + "-{epoch}-{val_pearson}",
    #                     monitor="val_pearson",
    #                     mode="max",
    #                     save_top_k=3,
    #                     )
    # ]
  

    # # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    # trainer = pl.Trainer(accelerator='gpu', max_epochs=args.max_epoch, logger=wandb_logger, log_every_n_steps=100, callbacks=checkpoint_callback)

    # # Train part
    # trainer.fit(model=model, datamodule=dataloader)
    # trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, './model/{model_name}/model_{version}.pt'.format(model_name=args.model_name.replace('/','_'),version=args.version))

    # 예측
    if args.write_output_file:
        predictions = trainer.predict(model=model, datamodule=dataloader)

        # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
        predictions = list(round(float(i), 1) for i in torch.cat(predictions))

        # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
        output = pd.read_csv('./data/sample_submission.csv')
        output['target'] = predictions
        output.to_csv(args.version + '_output.csv', index=False)

def train_kcontinue(args):
    wandb_logger = WandbLogger(project=args.project_name)
    cfg = {
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "epochs": args.max_epoch,
        "shuffle": args.shuffle,
        "learning_rate": args.learning_rate,
        "data_size" : 9874,
        "hidden_dropout": "0.1",
        "attention_dropout":"0.1",
        "ADAMW":"bias=(0.9,0.999),eps=1e-6",
        "loss function":"MSE(L2)",
    }
    wandb.config.update(cfg)

    # dataloader와 model을 생성합니다.
    dataloader = KfoldDataloader(args.model_name, args.batch_size, args.shuffle, 
                                 args.total_path, args.test_path, args.predict_path)
    model = torch.load('./model/{model_name}/model_{version}.pt'.format(model_name=args.model_name.replace('/','_'),version=args.version))

    checkpoint_callback = [
        EarlyStopping(monitor="val_pearson", min_delta=0.00, patience=5, verbose=False, mode="max",),
        ModelCheckpoint(
                        dirpath="./model/{model_name}/".format(model_name=args.model_name.replace('/','_')),
                        filename=args.model_name.replace('/','_') + "-{epoch}-{val_pearson}",
                        monitor="val_pearson",
                        mode="max",
                        save_top_k=3,
                        )
    ]

    trainer = pl.Trainer(accelerator='gpu', max_epochs=args.max_epoch, logger=wandb_logger, log_every_n_steps=100, callbacks=checkpoint_callback)

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, './model/{model_name}/model_cont_{version}.pt'.format(model_name=args.model_name.replace('/','_'),version=args.version))

    # 예측
    if args.write_output_file:
        predictions = trainer.predict(model=model, datamodule=dataloader)

        # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
        predictions = list(round(float(i), 1) for i in torch.cat(predictions))

        # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
        output = pd.read_csv('./data/sample_submission.csv')
        output['target'] = predictions
        output.to_csv(args.version + '_output.csv', index=False)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-small', type=str)# 'klue/roberta-large'
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_epoch', default=30, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='./data/train.csv')
    parser.add_argument('--dev_path', default='./data/dev.csv')
    parser.add_argument('--total_path', default='./data/total.csv')
    parser.add_argument('--test_path', default='./data/dev.csv')
    parser.add_argument('--predict_path', default='./data/test.csv')
    parser.add_argument('--version', default='temp',type=str)
    parser.add_argument('--write_output_file', default=True)
    parser.add_argument('--project_name', default="nlp1-small_model",type=str)
    parser.add_argument('--train_continue', type=str2bool, default=False)
    args = parser.parse_args()
    
    if args.train_continue:
        train_kcontinue(args)
    else:
        train_kfold(args)