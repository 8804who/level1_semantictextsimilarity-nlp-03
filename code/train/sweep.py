import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from model import Model
from dataloader import Dataloader


def sweep():
    project_name = 'KR-ELECTRA-discriminator-sweep'
    sweep_config = {
        'method': 'bayes',
        'parameters':{
            'lr':{
                'distribution': 'uniform',
                'min': 1e-6,
                'max': 1e-5
            }
        },
        'metric': {'name':'val_pearson', 'goal':'maximize'}      
    }
    
    def sweep_train(config=None):
        wandb.init(config=config)
        config = wandb.config
        
        # dataloader와 model을 생성합니다.
        dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                                args.test_path, args.predict_path)
        model = Model(args.model_name, args.learning_rate)
        wandb_logger = WandbLogger(project=project_name)
        
        checkpoint_callback = ModelCheckpoint(
            save_top_k=5,
            monitor="val_pearson",
            mode="max",
            dirpath="./model/{model_name}_sweep/".format(model_name=args.model_name.replace('/','_')), 
            filename=args.model_name.replace('/','_')+"_sweep-{epoch}-{val_pearson}",
        )
        
        trainer = pl.Trainer(accelerator='gpu', max_epochs=args.max_epoch, \
                            logger=wandb_logger, log_every_n_steps=1, callbacks=[checkpoint_callback])
        # Train part
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)
    
    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
    wandb.agent(sweep_id=sweep_id, function=sweep_train, count=args.count)

if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='snunlp/KR-ELECTRA-discriminator', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=3, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='./data/train.csv')
    parser.add_argument('--dev_path', default='./data/dev.csv')
    parser.add_argument('--test_path', default='./data/dev.csv')
    parser.add_argument('--predict_path', default='./data/test.csv')
    parser.add_argument('--count', default=3)
    args = parser.parse_args(args=[])

    # sweep 시작
    sweep()
