# evaluation
evaluation = dict(interval=48, metric='mAP')
# optimizer
optimizer = dict(type='Adam', lr=1.25e-4, weight_decay=0.0001) # lr 0.0025
optimizer_config = dict(grad_clip=None)
#optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    step=[48, 66])
runner = dict(type='EpochBasedRunner', max_epochs=84)
checkpoint_config = dict(interval=48)
