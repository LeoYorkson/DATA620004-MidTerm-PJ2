import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

with open('SparseRCNN/results/vis/scalars.json', 'r') as f:
    data = [json.loads(line) for line in f]

loss_data = []
map_data = []

for entry in data:
    if 'loss' in entry:
        loss_data.append(entry)
    elif 'pascal_voc/mAP' in entry:
        map_data.append(entry)

loss_df = pd.DataFrame(loss_data)
loss_df['step'] = loss_df['step'].astype(int)
loss_df['epoch'] = loss_df['epoch'].astype(int)

fig, axs = plt.subplots(2, 2, figsize=(18, 12), dpi=600)

for i, loss_type in enumerate(['cls', 'bbox', 'iou']):
    for stage in range(6):
        axs[i // 2, i % 2].plot(loss_df['step'], loss_df[f's{stage}.loss_{loss_type}'], label=f's{stage}', lw=2)

    axs[i // 2, i % 2].set_title(f'{loss_type.upper()} Loss per Stage', fontsize=16)
    axs[i // 2, i % 2].set_xlabel('Step', fontsize=14)
    axs[i // 2, i % 2].set_ylabel(f'{loss_type.upper()} Loss', fontsize=14)
    axs[i // 2, i % 2].legend(fontsize=12)

total_loss_df = pd.DataFrame()
total_loss_df['step'] = loss_df['step']
total_loss_df['total_loss'] = loss_df[['s0.loss_cls', 's0.loss_bbox', 's0.loss_iou']].sum(axis=1)
total_loss_df['cls_loss'] = loss_df[['s0.loss_cls', 's1.loss_cls', 's2.loss_cls', 's3.loss_cls', 's4.loss_cls', 's5.loss_cls']].sum(axis=1)
total_loss_df['bbox_loss'] = loss_df[['s0.loss_bbox', 's1.loss_bbox', 's2.loss_bbox', 's3.loss_bbox', 's4.loss_bbox', 's5.loss_bbox']].sum(axis=1)
total_loss_df['iou_loss'] = loss_df[['s0.loss_iou', 's1.loss_iou', 's2.loss_iou', 's3.loss_iou', 's4.loss_iou', 's5.loss_iou']].sum(axis=1)

axs[1, 1].stackplot(total_loss_df['step'], total_loss_df['cls_loss'], total_loss_df['bbox_loss'], total_loss_df['iou_loss'], 
                    labels=['cls_loss', 'bbox_loss', 'iou_loss'], alpha=0.6)

axs[1, 1].set_title('Total Loss (Stacked)', fontsize=16)
axs[1, 1].set_xlabel('Step', fontsize=14)
axs[1, 1].set_ylabel('Loss', fontsize=14)
axs[1, 1].legend(loc='upper left', fontsize=12)

fig.savefig('SparseRCNN/results/vis/loss_train.pdf', format='pdf')

map_df = pd.DataFrame(map_data)
map_df['epoch'] = map_df['step']

fig3, ax3 = plt.subplots(figsize=(10, 6), dpi=600)
ax3.plot(map_df['epoch'], map_df['pascal_voc/mAP'], label='mAP', color='r', lw=2)
ax3.set_title('mAP per Epoch', fontsize=16)
ax3.set_xlabel('Epoch', fontsize=14)
ax3.set_ylabel('mAP', fontsize=14)
ax3.legend(fontsize=12)

fig3.savefig('SparseRCNN/results/vis/mAP.pdf', format='pdf')

plt.tight_layout()
plt.show()
