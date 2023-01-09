import torch
from ltr.dataset import Lasot, MSCOCOSeq, Got10k, TrackingNet
from ltr.dataset.trdataset import Youtube_VOS, Saliency
from ltr.data import processing, sampler, LTRLoader
# import ltr.models.tracking.transt as transt_models
import ltr.models.tracking.SA_Tracker as SAT_models
from ltr import actors
# from ltr.trainers.transt_trainer import TransTLTRTrainer
from ltr.trainers.SA_Tracker_trainer import TransTLTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU

import os
# os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1, 2"


def run(settings):
    # Most common settings are assigned in the settings struct
    settings.local_rank = 0 # to adapt the ddp trainner
    settings.device = 'cuda'
    settings.description = 'TransT with default settings.'
    settings.batch_size = 256
    settings.num_workers = 32
    settings.multi_gpu = True
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 4.0
    settings.template_area_factor = 2.0
    settings.search_feature_sz = 32
    settings.template_feature_sz = 16
    settings.search_sz = settings.search_feature_sz * 8
    settings.temp_sz = settings.template_feature_sz * 8
    settings.center_jitter_factor = {'search': 3.0, 'static_template': 0, 'dynamic_template': 0.25}
    settings.scale_jitter_factor = {'search': 0.25, 'static_template': 0, 'dynamic_template': 0.25}

    #Backbone
    settings.backbone = 'MixFormer_Base'
    settings.embed_dim = 24
    settings.depths = [2, 4, 6, 6]
    settings.num_heads = [3, 6, 12, 24]
    settings.drop_path_rate = 0
    settings.position_embedding = 'sine'
    settings.hidden_dim = 192

    # Transformer
    # settings.position_embedding = 'sine'
    # settings.hidden_dim = 256
    # settings.dropout = 0.1
    # settings.nheads = 8
    # settings.dim_feedforward = 2048
    # settings.featurefusion_layers = 4

    # IOU head
    settings.iou_head = True
    # Segmentation
    settings.masks = False

    # pretrained_transt
    settings.freeze_transt = True
    settings.transt_path = '/data01/xlsun/code/TransT-M/results/SA_Tracker/checkpoints/ltr/SA_Tracker/SA_Tracker_dp/MixTracking_ep0496.pth.tar' 
    # settings.transt_path = None
    settings.env.workspace_dir = '/data01/xlsun/code/TransT-M/results/SA_Tracker_iouh_dp'
    settings.logdir = settings.env.workspace_dir + '/logs/'

    # Train datasets
    # -bbox
    lasot_train = Lasot(settings.env.lasot_dir, split='train')
    got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
    # trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(12)))
    coco_train = MSCOCOSeq(settings.env.coco_dir, version=2017)
    # -mask
    # youtube_vos = Youtube_VOS(settings.env.youtube_vos_dir)
    # saliency = Saliency(settings.env.saliency_dir)


    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # Data processing to do on the training pairs
    # data_processing_train = processing.TransTProcessing(search_area_factor=settings.search_area_factor,
    #                                                   template_area_factor = settings.template_area_factor,
    #                                                   search_sz=settings.search_sz,
    #                                                   temp_sz=settings.temp_sz,
    #                                                   center_jitter_factor=settings.center_jitter_factor,
    #                                                   scale_jitter_factor=settings.scale_jitter_factor,
    #                                                   mode='sequence',
    #                                                   transform=transform_train,
    #                                                   joint_transform=transform_joint)
    data_processing_train = processing.TransTMaskProcessing(search_area_factor=settings.search_area_factor,
                                                            template_area_factor = settings.template_area_factor,
                                                            search_sz=settings.search_sz,
                                                            temp_sz=settings.temp_sz,
                                                            center_jitter_factor=settings.center_jitter_factor,
                                                            scale_jitter_factor=settings.scale_jitter_factor,
                                                            mode='sequence',
                                                            transform=transform_train,
                                                            joint_transform=transform_joint)

    # The sampler for training
    # dataset_train = sampler.TransTMaskSampler([youtube_vos,saliency], [2,3],
    #                             samples_per_epoch=200*settings.batch_size, max_gap=100, processing=data_processing_train)
    # dataset_train = sampler.TransTMaskSampler([lasot_train, got10k_train, coco_train, trackingnet_train], [1,1,1,1],
    #                                           num_search_frames=1, num_template_frames=2,
    #                                           samples_per_epoch=200*settings.batch_size, max_gap=100, processing=data_processing_train,
    #                                           frame_sample_mode='transt')

    # dataset_train = sampler.TransTMaskSampler([got10k_train], [1],
    #                                         num_search_frames=1, num_template_frames=1,
    #                                         samples_per_epoch=200*settings.batch_size, max_gap=100, processing=data_processing_train,
    #                                         frame_sample_mode='transt')

    dataset_train = sampler.TransTMaskSampler([got10k_train, lasot_train, coco_train], [1, 1, 1],
                                        num_search_frames=1, num_template_frames=2,
                                        samples_per_epoch=300*settings.batch_size, max_gap=100, processing=data_processing_train,
                                        frame_sample_mode='transt')

    # The loader for training
    loader_train = LTRLoader('train', dataset_train, training=True,
                             batch_size=settings.batch_size,
                             num_workers=settings.num_workers,
                             drop_last=True, stack_dim=0, shuffle=True)

    # Create network and actor
    # model = transt_models.transt_resnet50(settings)
    model = SAT_models.SATracker(settings)

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        # model = MultiGPU(model, device_ids=[0, 1, 2], dim=0)
        model = MultiGPU(model, dim=0)

    objective = SAT_models.transt_loss(settings)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    actor = actors.TranstActor(net=model, objective=objective, settings=settings)

    # Optimizer
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
            # "lr": 1e-4,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-3,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=1e-3,
                                  weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50)

    # Create trainer
    trainer = TransTLTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(150, transt_path=settings.transt_path, load_latest=True, fail_safe=True)
