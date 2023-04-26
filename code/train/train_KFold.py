import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from dataloader import KFDataloader
from model import Model


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-small', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='./data/train.csv')
    parser.add_argument('--dev_path', default='./data/dev.csv')
    parser.add_argument('--test_path', default='./data/dev.csv')
    parser.add_argument('--predict_path', default='./data/test.csv')
    parser.add_argument('--warmup_ratio', default=0.1)
    parser.add_argument('--num_splits', default=5)
    args = parser.parse_args(args=[])

    # dataloader와 model을 생성합니다.
    model = Model(args.model_name, args.learning_rate)
    result = {}
    
    for fold in range(args.num_splits):
        print("--------------------------------{}fold start--------------------------------".format(fold))
        wandb_logger = WandbLogger(project="cross validation", name="{}th fold".format(fold))
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_pearson",
            mode="max",
            dirpath="./model/{model_name}/".format(model_name=args.model_name.replace('/','_') + '_cross-validation'), 
            filename=args.model_name.replace('/','_')+ f"_{fold}th fold" + "_-{epoch}-{val_pearson}",  # file 이름에 epoch이랑 val_pearson 같이 저장
        )
        dataloader = KFDataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path, args.num_splits, fold)

        # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
        trainer = pl.Trainer(accelerator='gpu', max_epochs=args.max_epoch, logger=wandb_logger, log_every_n_steps=1, callbacks=[checkpoint_callback])
        # Train part
        trainer.fit(model=model, datamodule=dataloader)
        score = trainer.test(model=model, datamodule=dataloader)
        wandb.finish()
        
        result[fold] = score

    # 모든 교차검증이 완료된 후 Test
    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, './model/{model_name}/model.pt'.format(model_name=args.model_name.replace('/','_') + '_cross-validation'))
    print(result)
