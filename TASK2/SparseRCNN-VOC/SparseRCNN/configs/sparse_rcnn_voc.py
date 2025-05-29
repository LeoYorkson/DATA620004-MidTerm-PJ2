_base_ = [
    '../../mmdetection/configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc.py',
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ]
)

train_dataloader = dict(batch_size=12, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=2)

optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
)

max_epochs = 12
train_cfg = dict(max_epochs=max_epochs)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=12)
)