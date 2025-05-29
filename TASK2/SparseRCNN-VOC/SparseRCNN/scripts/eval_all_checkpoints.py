import os
import json
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import init_default_scope
from torch.utils.tensorboard import SummaryWriter

import torch
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import MODELS
from mmdet.registry import DATASETS
from mmengine.evaluator import Evaluator
from tqdm import tqdm

log_dir = 'SparseRCNN/results/val_loss_curve'
writer = SummaryWriter(log_dir=log_dir)

cfg_path = 'SparseRCNN/results/20250528_223503/vis_data/config.py'
cfg = Config.fromfile(cfg_path)
cfg.work_dir = 'SparseRCNN/results'
init_default_scope(cfg.default_scope)

val_dataset = DATASETS.build(cfg.val_dataloader.dataset)
val_dataloader = Runner.build_dataloader(cfg.val_dataloader)

evaluator = Evaluator(cfg.val_evaluator)

def build_model(cfg):
    model = MODELS.build(cfg.model)
    model = revert_sync_batchnorm(model)
    return model

checkpoint_dir = cfg.work_dir
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('epoch_') and f.endswith('.pth')]
checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

all_scalars = {}

for ckpt_file in checkpoint_files:
    epoch_id = int(ckpt_file.split('_')[1].split('.')[0])
    ckpt_path = os.path.join(checkpoint_dir, ckpt_file)

    model = build_model(cfg)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    total_loss = 0.0
    loss_items_accumulator = {}
    num_batches = 0

    for data_batch in tqdm(val_dataloader, desc=f'Evaluating epoch {epoch_id}'):
        with torch.no_grad():
            data_batch = model.data_preprocessor(data_batch, training=False)
            outputs = model.forward(**data_batch, mode='loss')

            batch_loss = sum(v.item() for k, v in outputs.items() if 'loss' in k)
            total_loss += batch_loss

            for k, v in outputs.items():
                if k not in loss_items_accumulator:
                    loss_items_accumulator[k] = 0.0
                loss_items_accumulator[k] += v.item()

            num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"\n[Epoch {epoch_id}] Avg Total Loss: {avg_loss:.4f}")
    writer.add_scalar('Val/TotalLoss', avg_loss, epoch_id)

    epoch_scalars = {'TotalLoss': avg_loss}

    for k in loss_items_accumulator:
        loss_items_accumulator[k] /= num_batches
        print(f"  {k}: {loss_items_accumulator[k]:.4f}")
        writer.add_scalar(f'Val/{k}', loss_items_accumulator[k], epoch_id)
        epoch_scalars[k] = loss_items_accumulator[k]

    all_scalars[epoch_id] = epoch_scalars

writer.close()

scalars_path = os.path.join(log_dir, 'scalars.json')
with open(scalars_path, 'w') as f:
    json.dump(all_scalars, f, indent=2)
print(f"\nScalars saved to {scalars_path}")
