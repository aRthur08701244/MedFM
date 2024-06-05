_base_ = [
    '../_base_/models/densenet/densenet121-multilabel.py',
    '../_base_/datasets/imagenet_bs64-endo.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrain/densenet121_4xb256_in1k_20220426-07450f99.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=4))

train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)
