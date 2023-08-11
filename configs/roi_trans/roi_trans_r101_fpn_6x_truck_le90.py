_base_ = 'roi_trans_r50_fpn_6x_truck_le90.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')))
