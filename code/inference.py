import argparse
import pandas as pd

import torch
import pytorch_lightning as pl

import dataloader as data_loader
import model as MM

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
    parser.add_argument('--load_model_path', defualt=".pt",type=str)
    parser.add_argument('--additional_preprocessing', default=True, type=str2bool)
    args = parser.parse_args()


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
    
    dataloader = data_loader.Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path, args.additional_preprocessing)

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(accelerator='gpu', max_epochs=1, log_every_n_steps=100)

    if '.ckpt' in args.load_model_path:
        # ckpt = "snunlp_KR-ELECTRA-discriminator-epoch=14-val_pearson=0.9314203262329102"
        # state_dict = torch.load(f"./model/snunlp_KR-ELECTRA-discriminator/{ckpt}.ckpt")['state_dict']
        state_dict = torch.load(args.load_model_path)['state_dict']
        model = MM.Model(cfg)
        model.resize_token_embeddings(dataloader.len_tokenizer())
        model.load_state_dict(state_dict)

    
    else:  
        model = torch.load('./model/{model_name}/model.pt'.format(model_name=args.model_name.replace('/','_')))
   
    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(float(i) for i in torch.cat(predictions))
    # predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('./data/sample_submission.csv')
    output['target'] = predictions
    output.to_csv('output_person2.csv', index=False)