import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

with open('SparseRCNN/results/vis/scalars_val.json', 'r') as f:
    data = json.load(f)

epochs = list(data.keys())
total_loss = []
cls_loss = {f's{i}': [] for i in range(6)}
bbox_loss = {f's{i}': [] for i in range(6)}
iou_loss = {f's{i}': [] for i in range(6)}

for epoch in epochs:
    entry = data[epoch]
    total_loss.append(entry["TotalLoss"])
    
    for i in range(6):
        cls_loss[f's{i}'].append(entry.get(f's{i}.loss_cls', 0))
        bbox_loss[f's{i}'].append(entry.get(f's{i}.loss_bbox', 0))
        iou_loss[f's{i}'].append(entry.get(f's{i}.loss_iou', 0))

fig, axs = plt.subplots(2, 2, figsize=(18, 12), dpi=600)

for i, loss_type in enumerate(['cls', 'bbox', 'iou']):
    for stage in range(6):
        if loss_type == 'cls':
            axs[i // 2, i % 2].plot(epochs, cls_loss[f's{stage}'], label=f's{stage}', lw=2)
        elif loss_type == 'bbox':
            axs[i // 2, i % 2].plot(epochs, bbox_loss[f's{stage}'], label=f's{stage}', lw=2)
        else:
            axs[i // 2, i % 2].plot(epochs, iou_loss[f's{stage}'], label=f's{stage}', lw=2)

    axs[i // 2, i % 2].set_title(f'{loss_type.upper()} Loss per Stage', fontsize=16)
    axs[i // 2, i % 2].set_xlabel('Epoch', fontsize=14)
    axs[i // 2, i % 2].set_ylabel(f'{loss_type.upper()} Loss', fontsize=14)
    axs[i // 2, i % 2].legend(fontsize=12)

total_loss_df = pd.DataFrame()
total_loss_df['step'] = list(range(len(total_loss)))
total_loss_df['total_loss'] = total_loss
total_loss_df['cls_loss'] = [sum(x) for x in zip(*[cls_loss[f's{i}'] for i in range(6)])]
total_loss_df['bbox_loss'] = [sum(x) for x in zip(*[bbox_loss[f's{i}'] for i in range(6)])]
total_loss_df['iou_loss'] = [sum(x) for x in zip(*[iou_loss[f's{i}'] for i in range(6)])]

axs[1, 1].stackplot(total_loss_df['step'], total_loss_df['cls_loss'], total_loss_df['bbox_loss'], total_loss_df['iou_loss'], 
                    labels=['cls_loss', 'bbox_loss', 'iou_loss'], alpha=0.6)

axs[1, 1].set_title('Total Loss (Stacked)', fontsize=16)
axs[1, 1].set_xlabel('Epoch', fontsize=14)
axs[1, 1].set_ylabel('Loss', fontsize=14)
axs[1, 1].legend(loc='upper left', fontsize=12)

fig.savefig('SparseRCNN/results/vis/loss_validation.pdf', format='pdf')

plt.tight_layout()
plt.show()
