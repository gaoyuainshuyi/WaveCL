"""
Modified from DETR https://github.com/facebookresearch/detr
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter, UninitializedParameter
from models.backbone import build_backbone
from models.video_swin_transformer import build_video_swin_backbone
from models.matcher import build_matcher
from models.segmentation import FPNSpatialDecoder
from models.criterion import SetCriterion
from models.postprocessing import A2DSentencesPostProcess, ReferYoutubeVOSPostProcess, COCOPostProcess, PostProcess, PostProcessSegm
from models.position_encoding import PositionEmbeddingSine1D
from models.voc import VOC
from models.vla import MMF, FocusedLinearAttention
from transformers import RobertaModel, RobertaTokenizerFast, RobertaTokenizer
from einops import rearrange, repeat
from misc import NestedTensor, inverse_sigmoid
from .deformable_transformer import build_deforamble_transformer
import math
import copy
from typing import Dict
import os
import pywt
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from torch.cuda.amp import autocast
from linformer import Linformer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def wavelet_transform(x, filters):
    device = x.device
    filters = filters.to(device)
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_transform(x, filters):
    device = x.device
    filters = filters.to(device)
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

class VLP(nn.Linear):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, ind: bool = False, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(VLP, self).__init__(in_features, out_features)
        self.ind = ind
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.avg = nn.AdaptiveAvgPool2d([None,1])
        self.max = nn.AdaptiveMaxPool2d([None,1])
        # self.bn_s_w = nn.LayerNorm([self.out_features, self.in_features])
        self.bn_s_w = nn.BatchNorm1d(self.out_features)
        self.conv_t_w = nn.Conv2d(1, 1, (3, 3), padding=(1,1))
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, vid_embeds: Tensor, txt_embeds: Tensor) -> Tensor:
        if self.ind:
            return F.linear(txt_embeds, self.weight, self.bias)
        else:
            weight = self.weight
            thw, b, c = vid_embeds.shape
            vid_embeds = rearrange(vid_embeds, 'thw b c -> b c thw')
            spatial_context_avg = self.avg(vid_embeds).unsqueeze(1) # tb 1 c 1
            spatial_context_max = self.max(vid_embeds).unsqueeze(1) # tb 1 c 1
            spatial_context = spatial_context_avg + spatial_context_max
            # spatial_context = spatial_context_max
            spatial_context = spatial_context.squeeze(1) # tb o 1

            txt_embeds = rearrange(txt_embeds, 'tb s c -> tb c s')
            txt_context_avg = self.avg(txt_embeds).unsqueeze(1) # tb 1 c 1
            txt_context_max = self.max(txt_embeds).unsqueeze(1) # tb 1 c 1
            txt_context = txt_context_avg + txt_context_max
            # txt_context = txt_context_max
            txt_context = txt_context.squeeze(1) # tb in 1

            attn = torch.matmul(spatial_context, txt_context.permute(0,2,1)) # tb o in
            attn = self.bn_s_w(attn)
            attn = self.relu(attn).unsqueeze(1)
            attn = self.conv_t_w(attn).squeeze(1)
            attn = self.sig(attn)

            weight = attn * weight.unsqueeze(0)
            txt_embeds = torch.matmul(weight, txt_embeds)
            txt_embeds = rearrange(txt_embeds, 'bt c l -> l bt c')
            txt_embeds = txt_embeds + self.bias

            return txt_embeds

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

# Vision-context language-gated Projection module（VLP）
class SEP(nn.Module):
    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.vlp = VLP(input_feat_size, output_feat_size, ind=False, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, video_features, text_features):
        x = self.vlp(video_features, text_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output

class WCL(nn.Module):
    """ The main module of the Semantic-Assisted Object Cluster"""
    def __init__(self, config):
        """
        Parameters:
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         WCl can detect in a single image. In our paper we use 20 in all settings.
            mask_kernels_dim: dim of the segmentation kernels and of the feature maps outputted by the spatial decoder.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        if config.backbone in ["video-swin-t", "video-swin-s", "video-swin-b"]:
            self.backbone = build_video_swin_backbone(config)
        elif config.backbone in ["resnet50"]:
            self.backbone = build_backbone(config)
        
        self.num_feature_levels = config.DeformTransformer['num_feature_levels']
        d_model = config.DeformTransformer['d_model']
        self.num_queries = config.DeformTransformer['num_queries']
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.class_embed = nn.Linear(d_model, config.num_classes)
        self.rel_coord = config.rel_coord

        self.transformer = build_deforamble_transformer(config.DeformTransformer)

        if self.num_feature_levels > 1:
            num_backbone_outs = len(self.backbone.strides[-3:])
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone.num_channels[-3:][_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, d_model, kernel_size=1),
                    nn.GroupNorm(32, d_model),
                ))
            for _ in range(self.num_feature_levels - num_backbone_outs): # downsample 2x
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, d_model, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, d_model),
                ))
                in_channels = d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.backbone.num_channels[-3:][0], d_model, kernel_size=1),
                    nn.GroupNorm(32, d_model),
                )])
        
        # initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(config.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = self.transformer.decoder.num_layers
        if config.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        self.text_encoder = RobertaModel.from_pretrained(config.text_encoder_type)
        self.vlp_projs = [] 
        # self.dwts = []
        # self.idwts = []
        self.wt_filters = []
        self.iwt_filters = []
        self.wl_convs = []
        self.convs = []
        for idx in range(len(self.input_proj)):
            vlp_proj = SEP(input_feat_size=self.text_encoder.config.hidden_size, output_feat_size=d_model, dropout=0.1)
            # dwt = DWT1DForward(wave='haar', J=1)
            # idwt = DWT1DInverse(wave='haar')
            # wl_conv = nn.Conv2d(4*d_model, 4*d_model, kernel_size=3, padding=1)
            self.vlp_projs.append(vlp_proj)
            # self.dwts.append(dwt)
            # self.wl_convs.append(wl_conv)
            
        for idx in range(len(self.input_proj)-2):
            wt_filter, iwt_filter = create_wavelet_filter('db1', d_model, d_model, torch.float)
            wt_filter = nn.Parameter(wt_filter, requires_grad=False)
            iwt_filter = nn.Parameter(iwt_filter, requires_grad=False)
            conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
            self.wt_filters.append(wt_filter)
            self.iwt_filters.append(iwt_filter)
            self.convs.append(conv)
        self.vlp_projs = nn.ModuleList(self.vlp_projs)
        # self.dwts = nn.ModuleList(self.dwts)
        # self.wl_convs = nn.ModuleList(self.wl_convs)
        self.convs = nn.ModuleList(self.convs)
        # self.idwts = nn.ModuleList(self.idwts)
        # self.wt_filters = nn.ModuleList(self.wt_filters)
        # self.iwt_filters = nn.ModuleList(self.iwt_filters)

        # self.sentence_proj = FeatureResizer(
        #     input_feat_size = d_model,
        #     output_feat_size = d_model,
        #     dropout = 0.1,
        # )

        # self.caption_pooling = CaptionPooling(d_model)
        # self.text_encoder.pooler = None  # this pooler is never used, this is a hack to avoid DDP problems...
        self.tokenizer = RobertaTokenizerFast.from_pretrained(config.text_encoder_type)
        self.freeze_text_encoder = config.freeze_text_encoder
        if self.freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        
        self.text_pos = PositionEmbeddingSine1D(d_model, normalize=True)

        self.query_embed = nn.Embedding(self.num_queries, d_model) 
        #self.instance_kernels_head = MLP(d_model, d_model, output_dim=config.mask_kernels_dim, num_layers=3) #set some hyperparameter
        self.spatial_decoder = FPNSpatialDecoder(d_model, 2 * [d_model] + [self.backbone.num_channels[0]], config.mask_kernels_dim)
        self.voc = VOC(config.VOC)

        # self.vla = VisualLanguageALignment(d_model, d_model)
        self.vlf = MMF(d_model=d_model, nhead=8)
        self.wavelet_vlf_l = MMF(d_model=d_model, nhead=8)
        self.wavelet_vlf_h = FocusedLinearAttention(dim=d_model, num_heads=8)
        self.lvf = MMF(d_model=d_model, nhead=8)


        self.txt_proj = FeatureResizer(
            input_feat_size = self.text_encoder.config.hidden_size,
            output_feat_size = d_model,
            dropout = 0.1,
        )

        self.controller_layers = config.controller_layers
        self.in_channels = config.mask_kernels_dim
        self.dynamic_mask_channels = config.dynamic_mask_channels
        self.mask_out_stride = 4
        self.mask_feat_stride = 4

        weight_nums, bias_nums = [], []
        for l in range(self.controller_layers):
            if l == 0:
                if self.rel_coord:
                    weight_nums.append((self.in_channels + 2) * self.dynamic_mask_channels)
                else:
                    weight_nums.append(self.in_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
            elif l == self.controller_layers - 1:
                weight_nums.append(self.dynamic_mask_channels * 1) # output layer c -> 1
                bias_nums.append(1)
            else:
                weight_nums.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.controller = MLP(d_model, d_model, self.num_gen_params, 3)
        for layer in self.controller.layers:
            nn.init.zeros_(layer.bias)
            nn.init.xavier_uniform_(layer.weight) 
        # self.bbox_attention = MHAttentionMap(d_model, d_model, self.transformer.nhead, dropout=0)


        self.vl_loss = config.vl_loss
        # if self.vl_loss:
        #     self.is_referred_head = nn.Linear(d_model, 2)  # binary 'is referred?' prediction head for object queries
        self.aux_loss = config.aux_loss

    def forward_text(self, text_queries, device):
        tokenized_queries = self.tokenizer.batch_encode_plus(text_queries, padding='longest', return_tensors='pt')
        tokenized_queries = tokenized_queries.to(device)
        #with torch.inference_mode(mode=self.freeze_text_encoder):
        encoded_text = self.text_encoder(**tokenized_queries, output_hidden_states=True)
        # txt_memory = encoded_text
        # Transpose memory because pytorch's attention expects sequence first
        txt_memory = rearrange(encoded_text.last_hidden_state, 'b l c -> l b c')
        # txt_memory = self.txt_proj(txt_memory)  # change text embeddings dim to model dim
        text_sentence_feature = encoded_text.pooler_output
        text_sentence_feature = self.txt_proj(text_sentence_feature)
        # text_sentence_feature = None
        # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
        txt_pad_mask = tokenized_queries.attention_mask.ne(1).bool()  # [B, S] #0 for pad
        text_feature = NestedTensor(txt_memory, txt_pad_mask)
        return text_feature, text_sentence_feature
    

    def forward(self, samples: NestedTensor, valid_indices, text_queries, targets):
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: Batched frames of shape [time x batch_size x 3 x H x W]
               - samples.mask: A binary mask of shape [time x batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_cls": The reference prediction logits for all queries.
                                     Shape: [time x batch_size x num_queries x 2]
               - "pred_masks": The mask logits for all queries.
                               Shape: [time x batch_size x num_queries x H_mask x W_mask]
               - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                                dictionaries containing the two above keys for each decoder layer.
        """
        device = samples.tensors.device
        text_features, text_sentence_feature = self.forward_text(text_queries, device)
        backbone_out, pos = self.backbone(samples) #[backbone_out = [(b t) c h w]] mask: [(b t) h w]
        # keep only the valid frames (frames which are annotated):
        # (for example, in a2d-sentences only the center frame in each window is annotated).
        
        B = len(text_queries)
        BT = pos[0].shape[0]
        ## prepare for the deformable Transformer
        T = BT // B #a2d is one
        
        if valid_indices is not None:
            for layer_out in backbone_out:
                layer_out.tensors = layer_out.tensors.index_select(0, valid_indices) #[b*t c h w]
                layer_out.mask = layer_out.mask.index_select(0, valid_indices)
            for i, p in enumerate(pos):
                pos[i] = p.index_select(0, valid_indices) #[bt h w]
            samples.mask = samples.mask.index_select(0, valid_indices)
            T = 1 
        
        srcs = []
        langs = []
        masks = []
        poses = []

        text_pos = self.text_pos(text_features).permute(2, 0, 1)  # [length, batch_size, c]
        text_word_features, text_word_masks = text_features.decompose()  #text_word_feature [l b c]#text_word_mask [B L]
        # text_sentence_feature_fuse = text_sentence_feature.unsqueeze(0) #[1 b C]
        for l, (feat, pos_l) in enumerate(zip(backbone_out[-3:], pos[-3:])): 
            src, mask = feat.decompose()            
            src_proj_l = self.input_proj[l](src)    
            n, c, h, w = src_proj_l.shape
            text_feature = self.vlp_projs[l](rearrange(src_proj_l, '(b t) c h w -> (t h w) b c', b=B, t=T), text_word_features.permute(1,0,2))

            # text_feature_l, text_feature_h = self.dwts[l](text_feature.permute(1,2,0))
            # text_feature_l, text_feature_h = text_feature_l.permute(2,0,1), text_feature_h[0].permute(2,0,1) # [l/2 b c]
            if l < 2:
                video_proj = src_proj_l
                video_proj_shape = video_proj.shape
                if (video_proj_shape[2] % 2 > 0) or (video_proj_shape[3] % 2 > 0):
                    video_proj_pads = (0, video_proj_shape[3] % 2, 0, video_proj_shape[2] % 2)
                    video_proj = F.pad(src_proj_l, video_proj_pads)

                video_proj = wavelet_transform(video_proj, self.wt_filters[l])
                video_proj_ll = video_proj[:, :, 0, :, :]
                _, _, h_half, w_half = video_proj_ll.shape
                video_proj_h = video_proj[:, :, 1:4, :, :]

                # 2次wavelet
                video_proj_ll_shape = video_proj_ll.shape
                if (video_proj_ll_shape[2] % 2 > 0) or (video_proj_ll_shape[3] % 2 > 0):
                    video_proj_ll_pads = (0, video_proj_ll_shape[3] % 2, 0, video_proj_ll_shape[2] % 2)
                    video_proj_ll = F.pad(video_proj_ll, video_proj_ll_pads)
                video_proj_ll = wavelet_transform(video_proj_ll, self.wt_filters[l])
                video_proj_ll_ll = video_proj_ll[:, :, 0, :, :]
                video_proj_ll_h = video_proj_ll[:, :, 1:4, :, :]
                _, _, h_half_half, w_half_half = video_proj_ll_ll.shape
                video_proj_ll_ll = rearrange(video_proj_ll_ll, '(b t) c h w -> (t h w) b c', b=B, t=T)
                video_proj_ll_ll = self.wavelet_vlf_l(tgt=video_proj_ll_ll,
                                                 memory=text_sentence_feature.unsqueeze(0),
                                                 memory_key_padding_mask=None,
                                                 pos=None,
                                                 query_pos=None)
                video_proj_ll_h = rearrange(video_proj_ll_h, '(b t) c l h w -> (t l h w) b c', b=B, t=T)
                video_proj_ll_h = self.wavelet_vlf_h(x=video_proj_ll_h, memory=video_proj_ll_ll, H=h_half_half, W=w_half_half, T=T, l=3)
                video_proj_ll_ll = rearrange(video_proj_ll_ll, '(t h w) b c -> (b t) c h w', b=B, t=T, h=h_half_half, w=w_half_half)
                video_proj_ll_h = rearrange(video_proj_ll_h, '(t l h w) b c -> (b t) c l h w', b=B, t=T, h=h_half_half, w=w_half_half)
                video_proj_ll = torch.cat([video_proj_ll_ll.unsqueeze(2), video_proj_ll_h], dim=2)
                video_proj_ll = inverse_wavelet_transform(video_proj_ll, self.iwt_filters[l])[:, :, :h_half, :w_half]

                # 1次wavelet
                video_proj_h = rearrange(video_proj_h, '(b t) c l h w -> (t l h w) b c', b=B, t=T)
                video_proj_ll = rearrange(video_proj_ll, '(b t) c h w -> (t h w) b c', b=B, t=T)
                video_proj_ll = self.wavelet_vlf_l(tgt=video_proj_ll,
                                                 memory=text_sentence_feature.unsqueeze(0),
                                                 memory_key_padding_mask=None,
                                                 pos=None,
                                                 query_pos=None)
                video_proj_h = self.wavelet_vlf_h(x=video_proj_h, memory=video_proj_ll, H=h_half, W=w_half, T=T, l=3)
                video_proj_ll = rearrange(video_proj_ll, '(t h w) b c -> (b t) c h w', b=B, t=T, h=h_half, w=w_half)
                video_proj_h = rearrange(video_proj_h, '(t l h w) b c -> (b t) c l h w', b=B, t=T, h=h_half, w=w_half)
                video_proj = torch.cat([video_proj_ll.unsqueeze(2), video_proj_h], dim=2)
                # video_proj = rearrange(video_proj, '(b t) c l h w -> (b t) (c l) h w', b=B, t=T)
                # video_proj = F.leaky_relu(self.wl_convs[l](video_proj))
                # video_proj = rearrange(video_proj, '(b t) (c l) h w -> (b t) c l h w', b=B, t=T, l=4)
                video_proj = inverse_wavelet_transform(video_proj, self.iwt_filters[l])[:, :, :h, :w]
            # vision language early-fusion
            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> (t h w) b c', b=B, t=T)
            mask_l = rearrange(mask, '(b t) h w -> b (t h w)', t=T, b=B)
            pos = rearrange(pos_l, "(b t) c h w -> (t h w) b c", t=T, b=B)
            src_proj_l_new = self.vlf(tgt=src_proj_l,
                                             memory=text_feature,
                                             memory_key_padding_mask=text_word_masks,
                                             pos=text_pos,
                                             query_pos=None)

            lan_l = self.lvf(tgt=text_feature,
                             memory=src_proj_l,
                             memory_key_padding_mask = mask_l,
                            #  tgt_padding_mask = text_word_masks,
                             pos = pos,
                             query_pos = None
                             ) # src: [(t h w) b c] lan [l b c]
            # lan_l = lan_l * text_feature
            src_proj_l_new = rearrange(src_proj_l_new, '(t h w) b c -> (b t) c h w', t=T, h=h, w=w)
            if l < 2:
                src_proj_l_new = F.leaky_relu(self.convs[l](src_proj_l_new * torch.sigmoid(video_proj)))
            # else:
            # 	src_proj_l_new = src_proj_l_new * rearrange(src_proj_l, '(t h w) b c -> (b t) c h w', t=T, h=h, w=w)
            lan_l = rearrange(lan_l, 'l b c -> b l c')

            srcs.append(src_proj_l_new)
            masks.append(mask)
            poses.append(pos_l)
            langs.append(lan_l) #[b l c]
            assert mask is not None
            
        if self.num_feature_levels > (len(backbone_out) - 1):
            _len_srcs = len(backbone_out) - 1 # fpn level
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](backbone_out[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape
                text_feature = self.vlp_projs[l](rearrange(src, '(b t) c h w -> (t h w) b c', b=B, t=T), text_word_features.permute(1,0,2))

                # text_feature_l, text_feature_h = self.dwts[l](text_feature.permute(1,2,0))
                # text_feature_l, text_feature_h = text_feature_l.permute(2,0,1), text_feature_h[0].permute(2,0,1) # [l/2 b c]

                # video_proj = src
                # video_proj_shape = video_proj.shape
                # if (video_proj_shape[2] % 2 > 0) or (video_proj_shape[3] % 2 > 0):
                #     video_proj_pads = (0, video_proj_shape[3] % 2, 0, video_proj_shape[2] % 2)
                #     video_proj = F.pad(src, video_proj_pads)

                # video_proj = wavelet_transform(video_proj, self.wt_filters[l-1])
                # video_proj_ll = video_proj[:, :, 0, :, :]
                # _, _, h_half, w_half = video_proj_ll.shape
                # video_proj_h = video_proj[:, :, 1:4, :, :]

                # # 2次wavelet
                # video_proj_ll_shape = video_proj_ll.shape
                # if (video_proj_ll_shape[2] % 2 > 0) or (video_proj_ll_shape[3] % 2 > 0):
                #     video_proj_ll_pads = (0, video_proj_ll_shape[3] % 2, 0, video_proj_ll_shape[2] % 2)
                #     video_proj_ll = F.pad(video_proj_ll, video_proj_ll_pads)
                # video_proj_ll = wavelet_transform(video_proj_ll, self.wt_filters[l-1])
                # video_proj_ll_ll = video_proj_ll[:, :, 0, :, :]
                # video_proj_ll_h = video_proj_ll[:, :, 1:4, :, :]
                # _, _, h_half_half, w_half_half = video_proj_ll_ll.shape
                # video_proj_ll_ll = rearrange(video_proj_ll_ll, '(b t) c h w -> (t h w) b c', b=B, t=T)
                # video_proj_ll_ll = self.wavelet_vlf_l(tgt=video_proj_ll_ll,
                #                                  memory=text_sentence_feature.unsqueeze(0),
                #                                  memory_key_padding_mask=None,
                #                                  pos=None,
                #                                  query_pos=None)
                
                # video_proj_ll_h = rearrange(video_proj_ll_h, '(b t) c l h w -> (t l h w) b c', b=B, t=T)
                # video_proj_ll_h = self.wavelet_vlf_h(x=video_proj_ll_h, memory=video_proj_ll_ll, H=h_half_half, W=w_half_half, T=T, l=3)
                # video_proj_ll_ll = rearrange(video_proj_ll_ll, '(t h w) b c -> (b t) c h w', b=B, t=T, h=h_half_half, w=w_half_half)
                # video_proj_ll_h = rearrange(video_proj_ll_h, '(t l h w) b c -> (b t) c l h w', b=B, t=T, h=h_half_half, w=w_half_half)
                # video_proj_ll = torch.cat([video_proj_ll_ll.unsqueeze(2), video_proj_ll_h], dim=2)
                # video_proj_ll = inverse_wavelet_transform(video_proj_ll, self.iwt_filters[l-1])[:, :, :h_half, :w_half]

                # # 1次wavelet
                # video_proj_h = rearrange(video_proj_h, '(b t) c l h w -> (t l h w) b c', b=B, t=T)
                # video_proj_ll = rearrange(video_proj_ll, '(b t) c h w -> (t h w) b c', b=B, t=T)
                # video_proj_ll = self.wavelet_vlf_l(tgt=video_proj_ll,
                #                                  memory=text_sentence_feature.unsqueeze(0),
                #                                  memory_key_padding_mask=None,
                #                                  pos=None,
                #                                  query_pos=None)
                # video_proj_h = self.wavelet_vlf_h(x=video_proj_h, memory=video_proj_ll, H=h_half, W=w_half, T=T, l=3)
                # video_proj_ll = rearrange(video_proj_ll, '(t h w) b c -> (b t) c h w', b=B, t=T, h=h_half, w=w_half)
                # video_proj_h = rearrange(video_proj_h, '(t l h w) b c -> (b t) c l h w', b=B, t=T, h=h_half, w=w_half)
                # video_proj = torch.cat([video_proj_ll.unsqueeze(2), video_proj_h], dim=2)
                # # video_proj = rearrange(video_proj, '(b t) c l h w -> (b t) (c l) h w', b=B, t=T)
                # # video_proj = F.leaky_relu(self.wl_convs[l](video_proj))
                # # video_proj = rearrange(video_proj, '(b t) (c l) h w -> (b t) c l h w', b=B, t=T, l=4)
                # video_proj = inverse_wavelet_transform(video_proj, self.iwt_filters[l-1])[:, :, :h, :w]

                # vision language early-fusion
                src = rearrange(src, '(b t) c h w -> (t h w) b c', b=B, t=T)
                src = self.vlf(tgt=src,
                                memory=text_feature,
                                memory_key_padding_mask=text_word_masks,
                                pos=text_pos,
                                query_pos=None
                )
                src = rearrange(src, '(t h w) b c -> (b t) c h w', t=T, h=h, w=w)
                # src = F.leaky_relu(self.convs[l-1](src * torch.sigmoid(video_proj)))
                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)

        query_embeds = self.query_embed.weight #[num_queries, C]
        tgt = torch.zeros_like(query_embeds)
        tgt = repeat(tgt, 'nq c -> b t nq c', b=B, t=T)
        #text_embed = repeat(text_sentence_feature, 'b c -> b t q c', t=T, q=self.num_queries)
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, inter_samples = \
                                            self.transformer(srcs, tgt, masks, poses, query_embeds)
        # hs: [l, batch_size*time, num_queries_per_frame, c]
        # memory: list[Tensor], shape of tensor is [batch_size*time, c, hi, wi]
        # init_reference: [batch_size*time, num_queries_per_frame, 2]
        # inter_references: [l, batch_size*time, num_queries_per_frame, 4]

        layer_outputs = []
        # text_query = torch.zeros(1,B,256).to(device)
        if self.vl_loss:
            # text feature for vl_loss
            text_features = []
            tempt = []

            # for idx, lang in enumerate(langs): #list(B L C)
            lang = [t_mem[~pad_mask] for t_mem, pad_mask in zip(langs[-1], text_word_masks)] #[List B S C]
            for obj in lang:
                obj = torch.mean(obj, dim = 0)  #[C]
                tempt.append(obj)
            text_features.append(torch.stack(tempt, dim=0)) #[B C]
            # tempt = []
            text_features = torch.stack(text_features, dim=0)[0]  #[b, c] #without layer
            # text_features = langs[-1] #[b 1 C]

        hs = rearrange(hs, 'l (b t) q c -> l t b q c', t=T, b=B)
        # text_sentence_feature = self.caption_pooling(text_feature.permute(1,0,2))
        # text_sentence_feature = self.sentence_proj(text_sentence_feature)
        voc_hs = self.voc(hs, text_sentence_feature) # VOC_hs [L B NQ C]

        # residual 
        frame = hs.shape[1]
        hs_voc = repeat(voc_hs, 'l b n c -> l t b n c', t=frame)
        hs_voc = hs + hs_voc

        outputs_classes = []
        outputs_coords = []
        hs_voc = rearrange(hs_voc, 'l t b n c -> l (b t) n c')
        for lvl in range(hs_voc.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs_voc[lvl])
            tmp = self.bbox_embed[lvl](hs_voc[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid() # cxcywh, range in [0,1]
            outputs_coords.append(outputs_coord)
            outputs_classes.append(outputs_class)

        outputs_coord = torch.stack(outputs_coords)
        outputs_classes = torch.stack(outputs_classes)
        # rearrange
        outputs_coord = rearrange(outputs_coord, 'l (b t) q n -> l t b q n', b=B, t=T)
        outputs_classes = rearrange(outputs_classes, 'l (b t) q n -> l t b q n', b=B, t=T)
        hs_voc = rearrange(hs_voc, 'l (b t) n c -> l t b n c', b=B, t=T)
        memory = [rearrange(mem, '(b t) c h w -> (t b) c h w', b=B, t=T) for mem in memory]
        fpn_first_input = rearrange(backbone_out[0].tensors, '(b t) c h w -> (t b) c h w', b=B, t=T)
        memory.insert(0, fpn_first_input)
        decoded_frame_features = self.spatial_decoder(memory[-1], memory[:-1][::-1])
        # output masks is: [L, T, B, N, H_mask, W_mask] (h // 4, w // 4)
        # dynamic conv
        mask_features = rearrange(decoded_frame_features, '(t b) d h w -> b t d h w', t=T, b=B)
        outputs_seg_masks = []
        for lvl in range(hs_voc.shape[0]):
            dynamic_mask_head_params = self.controller(hs_voc[lvl])   # [batch_size*time, num_queries_per_frame, num_params]
            dynamic_mask_head_params = rearrange(dynamic_mask_head_params, 't b q n -> b (t q) n', b=B, t=T)
            lvl_references = inter_references[lvl, ..., :2]
            lvl_references = rearrange(lvl_references, '(b t) q n -> b (t q) n', b=B, t=T)
            outputs_seg_mask = self.dynamic_mask_with_coords(mask_features, dynamic_mask_head_params, lvl_references, targets)
            outputs_seg_mask = rearrange(outputs_seg_mask, 'b (t q) h w -> t b q h w', t=T)
            outputs_seg_masks.append(outputs_seg_mask)
        output_masks = torch.stack(outputs_seg_masks, dim=0) #[l t b q h w]
        # if self.vl_loss:
        #     outputs_is_referred = self.is_referred_head(hs_VOC)  # [L,T, B, N, 2]
        
            # print('shape',text_feature.shape)
            
            # text_feature = self.text_feature_head(text_query, text_feature.transpose(0,1), text_feature.transpose(0,1))[0]
            # text_feature = text_feature + text_query
            # import pdb
            # pdb.set_trace()
        if self.vl_loss:    
            for pm, plg, pir, pb in zip(output_masks, voc_hs, outputs_classes, outputs_coord):
                layer_out = {'pred_masks': pm,    #[t,b,n,h,w]
                            "pred_logit": plg,   #[b,n,c]
                            "pred_boxes": pb,
                            # "text_sentence_feature": text_sentence_feature
                            "text_sentence_feature": text_features,  #[b,1 c]
                            "pred_cls": pir #[t b nq K]
                            }
                layer_outputs.append(layer_out)
        else:
            for pm, pir in zip(output_masks, outputs_classes):
                layer_out = {'pred_masks': pm,
                            'pred_cls': pir,
                            }
                layer_outputs.append(layer_out)
        out = layer_outputs[-1]  # the output for the last decoder layer is used by default
        if self.aux_loss:
            out['aux_outputs'] = layer_outputs[:-1]
        return out

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def dynamic_mask_with_coords(self, mask_features, mask_head_params, reference_points, targets):
        """
        Add the relative coordinates to the mask_features channel dimension,
        and perform dynamic mask conv.

        Args:
            mask_features: [batch_size, time, c, h, w]
            mask_head_params: [batch_size, time * num_queries_per_frame, num_params]
            reference_points: [batch_size, time * num_queries_per_frame, 2], cxcy
            targets (list[dict]): length is batch size
                we need the key 'size' for computing location.
        Return:
            outputs_seg_mask: [batch_size, time * num_queries_per_frame, h, w]
        """
        device = mask_features.device
        b, t, c, h, w = mask_features.shape
        # this is the total query number in all frames
        _, num_queries = reference_points.shape[:2]  
        q = num_queries // t  # num_queries_per_frame

        # prepare reference points in image size (the size is input size to the model)
        new_reference_points = [] 
        for i in range(b):
            img_h, img_w = targets[0][i]['size']
            scale_f = torch.stack([img_w, img_h], dim=0) 
            tmp_reference_points = reference_points[i] * scale_f[None, :] 
            new_reference_points.append(tmp_reference_points)
        new_reference_points = torch.stack(new_reference_points, dim=0) 
        # [batch_size, time * num_queries_per_frame, 2], in image size
        reference_points = new_reference_points  

        # prepare the mask features
        if self.rel_coord:
            reference_points = rearrange(reference_points, 'b (t q) n -> b t q n', t=t, q=q) 
            locations = compute_locations(h, w, device=device, stride=self.mask_feat_stride) 
            relative_coords = reference_points.reshape(b, t, q, 1, 1, 2) - \
                                    locations.reshape(1, 1, 1, h, w, 2) # [batch_size, time, num_queries_per_frame, h, w, 2]
            relative_coords = relative_coords.permute(0, 1, 2, 5, 3, 4) # [batch_size, time, num_queries_per_frame, 2, h, w]

            # concat features
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w', q=q) # [batch_size, time, num_queries_per_frame, c, h, w]
            mask_features = torch.cat([mask_features, relative_coords], dim=3)
        else:
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w', q=q) # [batch_size, time, num_queries_per_frame, c, h, w]
        mask_features = mask_features.reshape(1, -1, h, w) 

        # parse dynamic params
        mask_head_params = mask_head_params.flatten(0, 1) 
        weights, biases = parse_dynamic_params(
            mask_head_params, self.dynamic_mask_channels,
            self.weight_nums, self.bias_nums
        )

        # dynamic mask conv
        mask_logits = self.mask_heads_forward(mask_features, weights, biases, mask_head_params.shape[0]) 
        mask_logits = mask_logits.reshape(-1, 1, h, w)

        # upsample predicted masks
        assert self.mask_feat_stride >= self.mask_out_stride
        assert self.mask_feat_stride % self.mask_out_stride == 0

        mask_logits = aligned_bilinear(mask_logits, int(self.mask_feat_stride / self.mask_out_stride))
        mask_logits = mask_logits.reshape(b, num_queries, mask_logits.shape[-2], mask_logits.shape[-1])

        return mask_logits  # [batch_size, time * num_queries_per_frame, h, w]

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits

def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4 
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, device, stride=1):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device)

    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1) #[hidden_dim ]
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        #[input_dim , hidden_dim, hidden_dim, hidden_dim] [hidden_dim hidden_dim hidden_dim output_dim]
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output

class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        """
        q the query: [t b n c]
        key: the last memory: [t b c h w]
        """
        q = rearrange(q, 't b nq c -> (t b) nq c')
        k = rearrange(k, 't b c h w -> (t b) c h w')
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights

def build(args):
    device = args.device
    model = WCL(args)
    matcher = build_matcher(args)
    weight_dict = {'loss_con': args.con_loss_coef,
                   'loss_dice': args.dice_loss_coef,
                   'loss_sigmoid_focal': args.sigmoid_focal_loss_coef,
                   'loss_cls': args.class_loss_coef,
                   'loss_bbox': args.box_loss_coef,
                   'loss_giou':args.giou_coef}
    
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.DeformTransformer['dec_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef, text_refer=args.vl_loss, num_classes=args.num_classes)
    criterion.to(device)

    postprocessor = build_postprocessors(args.dataset_name)
    
    return model, criterion, postprocessor

def build_postprocessors(dataset_name):
    if dataset_name == 'a2d_sentences' or dataset_name == 'jhmdb_sentences':
        postprocessors = A2DSentencesPostProcess()
    elif dataset_name == 'ref_youtube_vos' or dataset_name == 'joint':
        postprocessors = ReferYoutubeVOSPostProcess() 
        # for coco pretrain postprocessor
    elif "coco" in dataset_name:
        postprocessors = {"bbox": PostProcess(),
                          "segm": PostProcessSegm(threshold=0.5)
                          }
    elif dataset_name == 'davis':
        postprocessors = None
    return postprocessors

