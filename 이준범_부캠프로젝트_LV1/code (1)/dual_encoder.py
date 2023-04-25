import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
# import pytorch_lightning as pl
import lightning.pytorch as pl

import wandb
from pytorch_lightning.loggers import WandbLogger
from finetuning_scheduler import FinetuningScheduler




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


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    # def tokenizing(self, dataframe):
    #     data1 = []
    #     data2 = []
    #     for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
    #         # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
    #         # text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
    #         text1 = item['sentence_1']
    #         text2 = item['sentence_2']
    #         outputs1 = self.tokenizer(text1, add_special_tokens=True, padding='max_length', truncation=True)
    #         outputs2 = self.tokenizer(text2, add_special_tokens=True, padding='max_length', truncation=True)
    #         data1.append(outputs1['input_ids'])
    #         data2.append(outputs2['input_ids'])
            
    #     return data1, data2

    def listing(self, dataframe):
        data1 = []
        data2 = []
        for idx, item in dataframe.iterrows():
            text1 = item['sentence_1']
            text2 = item['sentence_2']
            data1.append(text1)
            data2.append(text2)
            
        return data1, data2

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs1, inputs2 = self.listing(data)

        return inputs1, inputs2, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs1, train_inputs2, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs1, val_inputs2, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset1 = Dataset(train_inputs1, train_targets)
            self.train_dataset2 = Dataset(train_inputs2, train_targets)
            
            self.val_dataset1 = Dataset(val_inputs1, val_targets)
            self.val_dataset2 = Dataset(val_inputs2, val_targets)
            
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs1,test_inputs2, test_targets = self.preprocessing(test_data)
            
            self.test_dataset1 = Dataset(test_inputs1, test_targets)
            self.test_dataset2 = Dataset(test_inputs2, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs1, predict_inputs2, predict_targets = self.preprocessing(predict_data)
            
            self.predict_dataset1 = Dataset(predict_inputs1, [])
            self.predict_dataset2 = Dataset(predict_inputs2, [])
            

    def train_dataloader(self):
        A = torch.utils.data.DataLoader(self.train_dataset1, batch_size=self.batch_size, shuffle=args.shuffle)
        B = torch.utils.data.DataLoader(self.train_dataset2, batch_size=self.batch_size, shuffle=args.shuffle)
        
        return A, B

    def val_dataloader(self):
        A = torch.utils.data.DataLoader(self.val_dataset1, batch_size=self.batch_size)
        B = torch.utils.data.DataLoader(self.val_dataset2, batch_size=self.batch_size)
        
        return A, B

    def test_dataloader(self):
        A = torch.utils.data.DataLoader(self.test_dataset1, batch_size=self.batch_size)
        B = torch.utils.data.DataLoader(self.test_dataset2, batch_size=self.batch_size)
        
        return A, B

    def predict_dataloader(self):
        A =  torch.utils.data.DataLoader(self.predict_dataset1, batch_size=self.batch_size)
        B =  torch.utils.data.DataLoader(self.predict_dataset2, batch_size=self.batch_size)
        
        return A, B


class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        
        # 사용할 모델을 호출합니다.
        # self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
        #     pretrained_model_name_or_path=model_name, num_labels=1, hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5)

        self.plm = SentenceTransformer('all-MiniLM-L6-v2')


        
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.L1Loss()

    def forward(self, x):
        # x = self.plm(x)['logits']
        embeddings1 = model.encode(sentences1, convert_to_tensor=True)
        embeddings2 = model.encode(sentences2, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
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
        optimizer = torch.optim.AdamW(self.parameters(), lr= self.lr)
        return optimizer


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-small', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='./data/train.csv')
    parser.add_argument('--dev_path', default='./data/dev.csv')
    parser.add_argument('--test_path', default='./data/dev.csv')
    parser.add_argument('--predict_path', default='./data/test.csv')

    # args = parser.parse_args(args=[])
    args = parser.parse_args()


    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
    
    
    model = Model(args.model_name, args.learning_rate)

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'

    wandb_logger = WandbLogger(project="basic", entity="g4012s")
  


    trainer = pl.Trainer(accelerator='gpu', max_epochs=args.max_epoch, log_every_n_steps=1, logger=wandb_logger)

    torch.cuda.empty_cache()

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)


    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, 'model.pt')
