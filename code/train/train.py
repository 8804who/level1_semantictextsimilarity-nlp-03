import argparse
import wandb
import pandas as pd

from pytorch_lightning.loggers import WandbLogger
import dataloader as data_loader
import model as MM
import utils 

import torch
import pytorch_lightning as pl



def train_normal(args):
    wandb_logger = WandbLogger(project=args.project_name)
    cfg = {
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "epochs": args.max_epoch,
        "shuffle": args.shuffle,
        "learning_rate": args.learning_rate,
        "data_size" : 9324, # train:9324, augmented:14937
        "hidden_dropout": "0.1",
        "attention_dropout":"0.1",
        "ADAMW":"bias=(0.9,0.999),eps=1e-6",
        "loss function":"MSE(L2)",
    }
    wandb.config.update(cfg)

    # dataloader와 model을 생성합니다.
    dataloader = data_loader.Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path, args.additional_preprocessing)
    model = MM.Model(cfg)
    model.resize_token_embeddings(dataloader.len_tokenizer())

    checkpoint_callback = [utils.best_save(model_name=args.model_name)]
    if args.use_scheduler: checkpoint_callback.append(utils.early_stop())

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(accelerator='gpu', max_epochs=args.max_epoch, logger=wandb_logger, log_every_n_steps=100, callbacks=checkpoint_callback)

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

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

def train_continue(args):
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
    dataloader = data_loader.Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                                args.test_path, args.predict_path, args.additional_preprocessing)
    model = torch.load('./model/{model_name}/model_{version}.pt'.format(model_name=args.model_name.replace('/','_'),version=args.version))
    
    checkpoint_callback = [utils.best_save(model_name=args.model_name)]
    if args.use_scheduler: checkpoint_callback.append(utils.early_stop())

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
    parser.add_argument('--model_name', default='snunlp/KR-ELECTRA-discriminator', type=str)
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
    parser.add_argument('--project_name', default="nlp1-electra_model",type=str)
    parser.add_argument('--train_continue', type=str2bool, default=False)
    parser.add_argument('--use_scheduler',type=str2bool, default=False)
    parser.add_argument('--additional_preprocessing',type=str2bool, default=True)
    args = parser.parse_args()
    
    if args.train_continue: train_continue(args)
    else: train_normal(args)

    
