# evaluation
evaluation = dict(interval=48, metric='mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3,
    step=[32, 44])
runner = dict(type='EpochBasedRunner', max_epochs=48)
checkpoint_config = dict(interval=12)