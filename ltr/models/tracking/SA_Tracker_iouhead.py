from ltr import model_constructor

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2, interpolate,
                       nested_tensor_from_tensor_list, accuracy)

from ltr.models.loss.matcher import build_matcher
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import collections.abc as container_abcs
from itertools import repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from typing import Optional
import copy
from ltr.models.neck.position_encoding import build_position_encoding2
from ltr.models.neck.featurefusion_network import build_CrossAttnention_network
from ltr.models.neck.convolutional_block_attention_module import build_CBAM_network

from ltr.models.tracking.transt_seg import (TransTsegm, dice_loss, sigmoid_focal_loss)
from ltr.models.tracking.transt_iouhead import TransTiouh
from ltr.models.tracking.transt_iouh_seg import TransTiouhsegm
# from lib.models.mixformer.position_encoding import build_position_encoding
# from ltr.models.tracking.SA_Tracker import MixTracking


class MixTrackingiouh(nn.Module):
    """ This is the base class for Transformer Tracking, whcih jointly perform feature extraction and interaction. """
    def __init__(self, model, hidden_dim, freeze_transt=False):
        """ Initializes the model.
        """
        super().__init__()
        num_classes = 1
        self.class_embed = model.class_embed
        self.bbox_embed = model.bbox_embed
        self.backbone = model.backbone

        if freeze_transt:
            for p in self.parameters():
                p.requires_grad_(False)
        
        hidden_dim = hidden_dim
        self.iou_embed = MLP(hidden_dim + 4, hidden_dim, 1, 3)

    def forward(self, search, templates):

        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        
        bs, n_t, c, h, w = templates.shape
        templates = templates.reshape(bs * n_t, c, h, w)

        if not isinstance(templates, NestedTensor):
            templates = nested_tensor_from_tensor(templates)
        
        mask = [templates.mask, search.mask]

        search_feature = self.backbone(templates.tensors, search.tensors, mask)

        outputs_class = self.class_embed(search_feature)
        outputs_coord = self.bbox_embed(search_feature).sigmoid()
        outputs_iouh = self.iou_embed(torch.cat((search_feature, outputs_coord), 3)).sigmoid()
      
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_iouh': outputs_iouh[-1]}
        return out
    
    def track(self, search, templates: list):
        pass


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


