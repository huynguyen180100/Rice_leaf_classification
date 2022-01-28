import argparse
import wandb

from pytorch_lightning import Trainer

from data import FlowerDataModule
from model import FlowerClassification


def log_test_prediction(table, data):
    for idx, (prob, pred, label, image_path) in enumerate(data):
        table.add_data(idx, wandb.Image(image_path), pred, label, prob)

def main(args):
    wandb.init(project="Flower Classification", name="log prediction")
    columns = ["id", "image", "pred", "label", "prob"]
    table = wandb.Table(columns=columns)

    dm = FlowerDataModule(data_dir=args.data_dir)
    class_to_idx = dm.val_dataloader().dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = FlowerClassification().load_from_checkpoint(checkpoint_path=args.ckpt_path)
    trainer = Trainer(gpus=args.gpus)
    results = trainer.predict(model=model, dataloaders=dm.val_dataloader())
    
    preds_list = []
    for batch in results:
        probs, preds, labels, image_paths = batch
        probs = probs.squeeze().tolist()
        preds, labels = preds.squeeze().tolist(), labels.squeeze().tolist()
        preds_list += [(prob, idx_to_class[pred], idx_to_class[label], path)
                        for prob, pred, label, path in zip(probs, preds, labels, image_paths)]
    
    log_test_prediction(table, preds_list)
    wandb.log({"test_predictions" : table})
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/Users/nguyenthaihuuhuy/Excercise/Pytorch Practice/data/flower_photos")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="/Users/nguyenthaihuuhuy/Excercise/Pytorch Practice/logs/mlp_mixer/model.ckpt")
    args = parser.parse_args()

    main(args)