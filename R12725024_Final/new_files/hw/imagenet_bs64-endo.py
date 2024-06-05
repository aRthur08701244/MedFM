# dataset settings
dataset_type = 'MultiLabelDataset'
data_preprocessor = dict(
    num_classes=4,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/medfmc/MedFMC/endo',
        ann_file='endo_train_20.pkl',
        data_prefix='images',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/medfmc/MedFMC/endo',
        ann_file='endo_val_20.pkl',
        data_prefix='images',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
# val_evaluator = dict(type='Accuracy', topk=(1, 5))
# val_evaluator = dict(type='AveragePrecision')
val_evaluator = [
     dict(type='AveragePrecision'),
     dict(type='MultiLabelMetric', average='macro'),  # class-wise mean
     dict(type='MultiLabelMetric', average='micro'),  # overall mean
   ]

# If you want standard test, please manually configure the test dataset
# test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/medfmc/MedFMC/endo',
        ann_file='endo_test_WithLabel.pkl',
        data_prefix='images',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = val_evaluator
