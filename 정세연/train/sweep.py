import argparse
import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torch
import utils
import dataloader as data_loader
import model as MM

def sweep(args):
    project_name = args.project_name

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

    sweep_config = {
        "method": "bayes",
        "parameters": {
            "lr": {
                "distribution": "uniform",
                "min": 1e-5,
                "max": 3e-5,
            },
        },
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 30,
            "s": 2,
        },
    }

    sweep_config["metric"] = {"name": "test_pearson", "goal": "maximize"}

    def sweep_train(config=None):
        wandb.init(config=config)
        config = wandb.config

        dataloader = dataloader = data_loader.Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path, args.additional_preprocessing)
        model = MM.Model(cfg)
        model.resize_token_embeddings(dataloader.len_tokenizer())

        wandb_logger = WandbLogger(project=project_name)
        save_path = f"sweep/{args.model_name}/sweep_id_{wandb.run.name}/"
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=args.max_epoch,
            logger=wandb_logger,
            log_every_n_steps=100,
            callbacks=[
                utils.early_stop(),
                utils.best_save(),
            ],
        )
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)
        trainer.save_checkpoint(save_path + "model.ckpt")
        # torch.save(model, save_path + "model.pt")

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project_name,
    )

    wandb.agent(sweep_id=sweep_id, function=sweep_train, count=args.experiment_time)

    
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
    parser.add_argument('--experiment_time', default=3,type=int)
    
    args = parser.parse_args()

    sweep(args)
