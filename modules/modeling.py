# # # modules/modeling.py
# # from __future__ import absolute_import, division, print_function, unicode_literals

# # import logging
# # import torch
# # from torch import nn
# # import numpy as np

# # from modules.until_module import PreTrainedModel, AllGather, CrossEn
# # from modules.module_cross import CrossModel, CrossConfig, Transformer_Text, Transformer_Gate
# # from modules.module_clip import CLIP, convert_weights
# # from torch.nn.utils import parametrize
# # import torch.nn.functional as F

# # logger = logging.getLogger(__name__)
# # allgather = AllGather.apply


# # def show_log(task_config, info):
# #     if task_config is None or task_config.local_rank == 0:
# #         logger.warning(info)


# # def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
# #     if hasattr(source_config, source_attr_name):
# #         if default_value is None or getattr(source_config, source_attr_name) != default_value:
# #             setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
# #             show_log(source_config, "Set {}.{}: {}.".format(target_name, target_attr_name,
# #                                                             getattr(target_config, target_attr_name)))
# #     return target_config


# # def check_attr(target_name, task_config):
# #     return hasattr(task_config, target_name) and task_config.__dict__[target_name]


# # class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
# #     def __init__(self, cross_config, *inputs, **kwargs):
# #         super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
# #         self.cross_config = cross_config
# #         self.clip = None
# #         self.cross = None

# #     @classmethod
# #     def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):
# #         task_config = kwargs.get("task_config", None)
# #         if task_config is not None:
# #             if not hasattr(task_config, "local_rank"):
# #                 task_config.__dict__["local_rank"] = 0
# #             elif task_config.local_rank == -1:
# #                 task_config.local_rank = 0

# #         if state_dict is None:
# #             state_dict = {}

# #         pretrained_clip_name = getattr(task_config, 'pretrained_clip_name', "ViT-B/32")
# #         clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
# #         for key, val in clip_state_dict.items():
# #             new_key = "clip." + key
# #             if new_key not in state_dict:
# #                 state_dict[new_key] = val.clone()

# #         cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size,
# #                                                  state_dict=None, task_config=task_config)
# #         model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

# #         # 3D patch init
# #         if model.linear_patch == "3d":
# #             contain_conv2 = any(key.find("visual.conv2.weight") > -1 for key in state_dict.keys())
# #             if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
# #                 cp_weight = state_dict["clip.visual.conv1.weight"].clone()
# #                 kernel_size = model.clip.visual.conv2.weight.size(2)
# #                 conv2_size = list(model.clip.visual.conv2.weight.size())

# #                 left_conv2_size = conv2_size.copy()
# #                 right_conv2_size = conv2_size.copy()
# #                 left_conv2_size[2] = (kernel_size - 1) // 2
# #                 right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

# #                 left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device) if left_conv2_size[2] > 0 else None
# #                 right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device) if right_conv2_size[2] > 0 else None

# #                 cat_list = []
# #                 if left_zeros is not None:
# #                     cat_list.append(left_zeros)
# #                 cat_list.append(cp_weight.unsqueeze(2))
# #                 if right_zeros is not None:
# #                     cat_list.append(right_zeros)
# #                 cp_weight = torch.cat(cat_list, dim=2)
# #                 state_dict["clip.visual.conv2.weight"] = cp_weight

# #         # tightTransf init
# #         if model.sim_header == 'tightTransf':
# #             contain_cross = any(key.find("cross.transformer") > -1 for key in state_dict.keys())
# #             if contain_cross is False:
# #                 for key, val in clip_state_dict.items():
# #                     if key == "positional_embedding":
# #                         state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
# #                         continue
# #                     if key.find("transformer.resblocks") == 0:
# #                         num_layer = int(key.split(".")[2])
# #                         if num_layer < task_config.cross_num_hidden_layers:
# #                             state_dict["cross." + key] = val.clone()
# #                             continue

# #         # seq(LSTM/Transf) init
# #         if model.sim_header in ["seqLSTM", "seqTransf"]:
# #             contain_frame_position = any(key.find("frame_position_embeddings") > -1 for key in state_dict.keys())
# #             if contain_frame_position is False:
# #                 for key, val in clip_state_dict.items():
# #                     if key == "positional_embedding":
# #                         state_dict["frame_position_embeddings.weight"] = val.clone()
# #                         continue
# #                     if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
# #                         num_layer = int(key.split(".")[2])
# #                         if num_layer < task_config.cross_num_hidden_layers:
# #                             state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
# #                         if num_layer < task_config.audio_query_layers:
# #                             state_dict[key.replace("transformer.", "transformer_Fusion.")] = val.clone()
# #                             continue

# #         if state_dict is not None:
# #             model = cls.init_preweight(model, state_dict, task_config=task_config)
# #         return model


# # class CLIP4Clip(CLIP4ClipPreTrainedModel):
# #     def __init__(self, cross_config, clip_state_dict, task_config):
# #         super(CLIP4Clip, self).__init__(cross_config)
# #         self.task_config = task_config
# #         self.ignore_video_index = -1

# #         assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings
# #         self._stage_one = True
# #         self._stage_two = False
# #         show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

# #         self.loose_type = False
# #         if self._stage_one and check_attr('loose_type', self.task_config):
# #             self.loose_type = True
# #             show_log(task_config, "Test retrieval by loose type.")

# #         vit = "visual.proj" in clip_state_dict
# #         assert vit
# #         vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
# #         vision_layers = len([k for k in clip_state_dict.keys()
# #                              if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
# #         vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
# #         grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
# #         image_resolution = vision_patch_size * grid_size

# #         embed_dim = clip_state_dict["text_projection"].shape[1]
# #         context_length = clip_state_dict["positional_embedding"].shape[0]
# #         vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
# #         transformer_width = clip_state_dict["ln_final.weight"].shape[0]
# #         transformer_heads = transformer_width // 64
# #         transformer_layers = len({k.split(".")[2] for k in clip_state_dict.keys() if k.startswith("transformer.resblocks.")})

# #         self.linear_patch = getattr(task_config, "linear_patch", "2d")
# #         self.hidden_size = cross_config.hidden_size

# #         # Build CLIP
# #         cut_top_layer = 0
# #         self.clip = CLIP(
# #             embed_dim,
# #             image_resolution, vision_layers - cut_top_layer, vision_width, vision_patch_size,
# #             context_length, vocab_size, transformer_width, transformer_heads, transformer_layers - cut_top_layer,
# #             linear_patch=self.linear_patch
# #         ).float()
# #         for key in ["input_resolution", "context_length", "vocab_size"]:
# #             if key in clip_state_dict:
# #                 del clip_state_dict[key]
# #         convert_weights(self.clip)

# #         # args
# #         self.sim_header = getattr(self.task_config, "sim_header", "seqTransf")
# #         self.lambda_ = self.task_config.temperature

# #         # **단일 스위치**: retrieval_mode만 사용
# #         self.retrieval_mode = getattr(self.task_config, "retrieval_mode", "textquery_fuse")

# #         self.v_cross_layers = getattr(self.task_config, "v_cross_layers", 2)
# #         self.a_cross_layers = getattr(self.task_config, "a_cross_layers", 2)

# #         # modules
# #         num_query_token = 12
# #         self.query_tokens = nn.Parameter(torch.zeros(cross_config.hidden_size, num_query_token))
# #         nn.init.orthogonal_(self.query_tokens, 1.0)

# #         if self.loose_type is False:
# #             cross_config.max_position_embeddings = context_length
# #             cross_config = update_attr("cross_config", cross_config, "num_hidden_layers",
# #                                        self.task_config, "cross_num_hidden_layers")
# #             self.cross = CrossModel(cross_config)
# #             self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

# #         if self.sim_header in ["seqLSTM", "seqTransf"]:
# #             self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
# #             self.query_position_embeddings = nn.Embedding(num_query_token, cross_config.hidden_size)

# #         if self.sim_header == "seqTransf":
# #             # 기존 A-resampler + V-gated fusion
# #             self.transformer_Fusion = Transformer_Text(width=transformer_width,
# #                                                        layers=self.task_config.audio_query_layers,
# #                                                        heads=transformer_heads)
# #             self.transformerClip = Transformer_Gate(width=transformer_width,
# #                                                     layers=self.task_config.cross_num_hidden_layers,
# #                                                     heads=transformer_heads)
# #             if self.retrieval_mode == "five_weighted":
# #                 self.five_gate = nn.Sequential(
# #                     nn.Linear(self.hidden_size * 5, self.hidden_size),
# #                     nn.ReLU(inplace=True),
# #                     nn.Linear(self.hidden_size, 5),
# #                     nn.Sigmoid()
# #                 )

# #             if self.retrieval_mode in {
# #                     "five_weighted_without_obj",
# #                     "five_weighted_without_act",
# #                     "five_weighted_without_att",
# #                     "five_weighted_without_audm",
# #                 }:
# #                 self.five_gate = nn.Sequential(
# #                     nn.Linear(self.hidden_size * 4, self.hidden_size),
# #                     nn.ReLU(inplace=True),
# #                     nn.Linear(self.hidden_size, 4),
# #                     nn.Sigmoid()
# #                 )
# #             # --- ADDED: five_weighted_independent (Sigmoid-gated, no softmax) ---
# #             if self.retrieval_mode == "five_weighted_independent":
# #                 # (a) 슬롯별 av–text 페어 게이트: [av|t_i] -> (α_i, β_i) in [0,1]^2
# #                 self.five_pair_gate = nn.Sequential(
# #                     nn.Linear(self.hidden_size * 2, self.hidden_size),
# #                     nn.ReLU(inplace=True),
# #                     nn.Linear(self.hidden_size, 2),
# #                     nn.Sigmoid()  # 독립 게이트(합=1 강제 안 함)
# #                 )
# #                 # (b) 4개 후보 혼합 게이트: [cand_obj|cand_act|cand_att|cand_audm] -> γ in [0,1]^4
# #                 self.five_mix_gate = nn.Sequential(
# #                     nn.Linear(self.hidden_size * 4, self.hidden_size),
# #                     nn.ReLU(inplace=True),
# #                     nn.Linear(self.hidden_size, 4),
# #                     nn.Sigmoid()  # 독립 게이트
# #                 )

# #             # --- ADDED: self-attn over [av, t4, 3*learnable], mean pooling ---
# #             if self.retrieval_mode == "five_selfattn_mean":
# #                 # learnable 3 tokens (slot-level self-attn에서 정보 모음/분산용)
# #                 self.five_sa_extra = nn.Parameter(torch.zeros(3, self.hidden_size))
# #                 nn.init.orthogonal_(self.five_sa_extra, 1.0)

# #                 # 작은 층 수로 시작 (필요하면 task_config.five_sa_layers로 조절)
# #                 self.transformer_FiveSA = Transformer_Text(
# #                     width=transformer_width,
# #                     layers=getattr(self.task_config, "five_sa_layers", 1),
# #                     heads=transformer_heads,
# #                 )

# #                 # 풀링 시 learnable 토큰 포함 여부 (기본 포함)
# #                 self.five_sa_pool_include_extra = getattr(self.task_config, "five_sa_pool_include_extra", True)



# #             if self.retrieval_mode == "five_crossattn":
# #                 self.five_query_extra = nn.Parameter(torch.zeros(3, self.hidden_size))
# #                 nn.init.orthogonal_(self.five_query_extra, 1.0)
# #                 # cross-attn 레이어 (query: AV+3tok, key/value: [t_obj, t_act, t_attr, t_aud])
# #                 self.transformer_Five = Transformer_Text(
# #                     width=transformer_width,
# #                     layers=getattr(self.task_config, "five_query_layers", 1),
# #                     heads=transformer_heads
# #                 )

# #             if self.retrieval_mode == "av_disentangle_slots_proj":
# #                 self.disent_gate_simple = nn.Sequential(
# #                     nn.Linear(self.hidden_size * 3, self.hidden_size),
# #                     nn.ReLU(inplace=True),
# #                     nn.Linear(self.hidden_size, 2),
# #                     nn.Sigmoid()
# #                 )
# #             # --- ADDED: pair → attention 2-stage (Sigmoid gates, then self-attn + mean) ---
# #             if self.retrieval_mode == "five_pairattn_mean":
# #                 # 1단계: 슬롯별 av–text 페어 게이트 (공유)
# #                 self.five_pair_gate = nn.Sequential(
# #                     nn.Linear(self.hidden_size * 2, self.hidden_size),
# #                     nn.ReLU(inplace=True),
# #                     nn.Linear(self.hidden_size, 2),
# #                     nn.Sigmoid()  # 독립 게이트 (합=1 강제 안 함)
# #                 )
# #                 # 2단계: self-attn용 learnable 3 tokens
# #                 self.five_pairattn_extra = nn.Parameter(torch.zeros(3, self.hidden_size))
# #                 nn.init.orthogonal_(self.five_pairattn_extra, 1.0)

# #                 # 2단계: 후보 4개(+extra 3)를 한 번의 self-attn으로 통과
# #                 self.transformer_FivePairAttn = Transformer_Text(
# #                     width=transformer_width,
# #                     layers=getattr(self.task_config, "five_pairattn_layers", 1),
# #                     heads=transformer_heads,
# #                 )
# #                 # 풀링 시 extra 토큰 포함 여부 (기본 True)
# #                 self.five_pairattn_pool_include_extra = getattr(self.task_config, "five_pairattn_pool_include_extra", True)


# #             # __init__ 내, retrieval_mode 설정/모듈 정의들 아래 적당한 곳에 추가
# #             if self.retrieval_mode in ["slerp1", "slerp2"]:
# #                 # slerp1: tau를 학습 파라미터로 둘지, 고정할지 선택 가능
# #                 slerp1_tau_init = getattr(self.task_config, "slerp1_tau_init", 0.5)
# #                 slerp1_tau_learn = getattr(self.task_config, "slerp1_tau_learnable", True)
# #                 if slerp1_tau_learn:
# #                     self.slerp1_tau = nn.Parameter(torch.tensor(float(slerp1_tau_init)))
# #                 else:
# #                     self.register_buffer("slerp1_tau", torch.tensor(float(slerp1_tau_init)), persistent=False)

# #                 # slerp2: (1) AV–각 슬롯 slerp의 tau 4개, (2) 4개 결과를 섞는 가중치(softmax)
# #                 slerp2_tau_init = getattr(self.task_config, "slerp2_tau_init", 0.5)
# #                 slerp2_tau_learn = getattr(self.task_config, "slerp2_tau_learnable", True)
# #                 if slerp2_tau_learn:
# #                     self.slerp2_tau = nn.Parameter(torch.full((4,), float(slerp2_tau_init)))
# #                 else:
# #                     self.register_buffer("slerp2_tau", torch.full((4,), float(slerp2_tau_init)), persistent=False)

# #                 # 4개 결과의 구면 평균 가중치(학습)
# #                 w0 = getattr(self.task_config, "slerp2_weight_init", [1.0, 1.0, 1.0, 1.0])
# #                 w0 = torch.tensor(w0, dtype=torch.float)
# #                 assert w0.numel() == 4, "slerp2_weight_init must have 4 values"
# #                 self.slerp2_weight_logits = nn.Parameter(w0)


# #         if self.sim_header == "seqLSTM":
# #             self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size,
# #                                        hidden_size=cross_config.hidden_size,
# #                                        batch_first=True, bidirectional=False, num_layers=1)

# #         self.loss_fct_vis = CrossEn()
# #         self.apply(self.init_weights)

# #     # 클래스 내부에 추가
# #     def _five_pairattn_mean(self, av_g, obj_g, act_g, att_g, audm_g, include_extra=True):
# #         """
# #         Stage-1: cand_i = α_i * av + β_i * t_i   (i ∈ {obj,act,att,audm}, Sigmoid pair gates)
# #         Stage-2: tokens = [cand_obj, cand_act, cand_att, cand_audm, L1, L2, L3]
# #                 → one-shot self-attn (Q=K=V)
# #                 → masked mean pooling → q
# #         반환:
# #         q: [B, D]
# #         weights: {"pair":[B,4,2]}  # (α_i, β_i) per slot
# #         """
# #         B, D = av_g.size()
# #         slots = [obj_g, act_g, att_g, audm_g]

# #         # ---- Stage-1: per-slot pair mixing ----
# #         pair_ws = []  # [B,4,2]
# #         cands = []    # [B,4,D]
# #         for t in slots:
# #             feat = torch.cat([av_g, t], dim=-1)    # [B, 2D]
# #             w2 = self.five_pair_gate(feat)         # [B, 2] (Sigmoid)
# #             cand = w2[:, :1] * av_g + w2[:, 1:] * t
# #             cands.append(cand)
# #             pair_ws.append(w2)

# #         C = torch.stack(cands, dim=1)              # [B, 4, D]
# #         pair_ws = torch.stack(pair_ws, dim=1)      # [B, 4, 2]

# #         # ---- Stage-2: self-attn over candidates (+ 3 learnable tokens) ----
# #         extra = self.five_pairattn_extra.unsqueeze(0).expand(B, -1, -1)  # [B,3,D]
# #         tokens = torch.cat([C, extra], dim=1)       # [B, 7, D] = 4 cand + 3 extra

# #         # 풀링 마스크: 텍스트가 비어있는 cand는 제외, extra는 include_extra에 따라 포함
# #         txt = [obj_g, act_g, att_g, audm_g]
# #         txt_valid = [(t.norm(dim=-1) > 0).float() for t in txt]  # 각 [B]
# #         cand_mask = torch.stack(txt_valid, dim=1)                # [B,4]
# #         if include_extra:
# #             extra_mask = torch.ones(B, 3, device=av_g.device, dtype=av_g.dtype)
# #         else:
# #             extra_mask = torch.zeros(B, 3, device=av_g.device, dtype=av_g.dtype)
# #         pool_mask = torch.cat([cand_mask, extra_mask], dim=1)    # [B,7]

# #         # time-major로 변환 후 self-attn (Q=K=V 동일)
# #         x = tokens.permute(1, 0, 2)                              # [7, B, D]
# #         y = self.transformer_FivePairAttn(x, x, None)            # mask=None → 전체 유효
# #         y = y.permute(1, 0, 2).contiguous()                      # [B, 7, D]

# #         # 안전 가드: 모든 cand가 비어있고 extra도 제외된 배치가 있으면, pool_mask를 전체 1로 대체
# #         denom = pool_mask.sum(dim=1, keepdim=True)              # [B,1]
# #         need_fallback = (denom.squeeze(-1) == 0)
# #         if need_fallback.any():
# #             pool_mask = pool_mask.clone()
# #             pool_mask[need_fallback] = 1.0
# #             denom = pool_mask.sum(dim=1, keepdim=True)

# #         # masked mean pool
# #         m = pool_mask.unsqueeze(-1)                              # [B,7,1]
# #         q = (y * m).sum(dim=1) / denom.clamp(min=1.0)           # [B,D]

# #         # L2 정규화
# #         q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)

# #         weights = {"pair": pair_ws}
# #         return q, weights

# #     def _five_weighted_independent(self, av_g, obj_g, act_g, att_g, audm_g):
# #         """
# #         2-stage Sigmoid gating (합=1 비강제):
# #         1) cand_i = α_i * av + β_i * t_i,  i ∈ {obj,act,att,audm}
# #         2) q = Σ_i γ_i * cand_i
# #         return:
# #         q: [B, D]
# #         weights: {"pair":[B,4,2], "mix":[B,4]}
# #         """
# #         B, D = av_g.size()
# #         slots = [obj_g, act_g, att_g, audm_g]

# #         pair_ws = []   # [B,4,2]
# #         cands = []     # [B,4,D]

# #         # (1) 슬롯별 av–text 혼합
# #         for t in slots:
# #             feat = torch.cat([av_g, t], dim=-1)  # [B, 2D]
# #             w2 = self.five_pair_gate(feat)       # [B, 2], Sigmoid
# #             cand = w2[:, :1] * av_g + w2[:, 1:] * t   # [B, D]
# #             cands.append(cand)
# #             pair_ws.append(w2)

# #         C = torch.stack(cands, dim=1)           # [B, 4, D]
# #         pair_ws = torch.stack(pair_ws, dim=1)   # [B, 4, 2]

# #         # (2) 4 후보 혼합
# #         mix_feat = torch.cat([C[:, i, :] for i in range(4)], dim=-1)  # [B, 4D]
# #         w4 = self.five_mix_gate(mix_feat)        # [B, 4], Sigmoid
# #         q = (w4.unsqueeze(-1) * C).sum(dim=1)    # [B, D]

# #         # 최종 정규화
# #         q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)

# #         weights = {"pair": pair_ws, "mix": w4}
# #         return q, weights

# #     def _five_weighted(self, av_g, obj_g, act_g, att_g, audm_g):

# #         B, D = av_g.size()
# #         cat = torch.cat([av_g, obj_g, act_g, att_g, audm_g], dim=-1)      # [B, 5D]
# #         w = self.five_gate(cat)                                           # [B, 5], sigmoid로 0~1

# #         comps = torch.stack([av_g, obj_g, act_g, att_g, audm_g], dim=1)   # [B, 5, D]
# #         q = torch.sum(w.unsqueeze(-1) * comps, dim=1)                     # [B, D]

# #         q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #         return q, w

# #     def _five_weighted_without_one(self, av_g, t1, t2, t3):

# #         B, D = av_g.size()
# #         cat = torch.cat([av_g, t1, t2, t3], dim=-1)      # [B, 5D]
# #         w = self.five_gate(cat)                                           # [B, 5], sigmoid로 0~1

# #         comps = torch.stack([av_g, t1, t2, t3], dim=1)   # [B, 5, D]
# #         q = torch.sum(w.unsqueeze(-1) * comps, dim=1)                     # [B, D]

# #         q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #         return q, w


# #     def _five_crossattn(self, av_g, obj_g, act_g, att_g, audm_g):
# #         """
# #         AV를 쿼리로, (t_obj,t_act,t_attr,t_aud) 를 KV로 하는 cross-attn.
# #         쿼리에 learnable 3 tokens를 추가(총 4 query). 첫 쿼리 토큰(AV 자리)의 출력을 최종 q로 사용.
# #         return:
# #         q: [B, D]  (cross-attn 결과의 첫 query 토큰)
# #         """
# #         B, D = av_g.size()

# #         # Query: [B, 1+3, D]  (첫 토큰은 av_g, 나머지 3개는 학습 토큰)
# #         extra = self.five_query_extra.unsqueeze(0).expand(B, -1, -1)  # [B, 3, D]
# #         q_tokens = torch.cat([av_g.unsqueeze(1), extra], dim=1)       # [B, 4, D]

# #         # Key/Value: [B, 4, D] (obj, act, att, audm)
# #         kv_tokens = torch.stack([obj_g, act_g, att_g, audm_g], dim=1) # [B, 4, D]

# #         # 마스크(모두 유효)
# #         q_mask = torch.ones(B, q_tokens.size(1), device=av_g.device)
# #         kv_mask = torch.ones(B, kv_tokens.size(1), device=av_g.device)
# #         extended_q_mask = (1.0 - q_mask.unsqueeze(-1)) * -1000000.0
# #         extended_q_mask = extended_q_mask.expand(-1, -1, kv_tokens.size(1))  # [B, 4, 4]

# #         # position embedding은 사용 안 함(간단)
# #         q_in = q_tokens.permute(1, 0, 2)   # [Q,B,D]
# #         kv_in = kv_tokens.permute(1, 0, 2) # [K,B,D]

# #         q_out = self.transformer_Five(q_in, kv_in, extended_q_mask)   # [Q,B,D] 유사 인터페이스
# #         q_out = q_out.permute(1, 0, 2).contiguous()                   # [B,Q,D]

# #         # 첫 쿼리 토큰(AV 자리)만 취함
# #         q = q_out[:, 0, :]                                            # [B,D]
# #         q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #         return q

# #     # ==== helpers ====
# #     def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
# #         # sequence_output: [B, L, D], attention_mask: [B, L]
# #         attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
# #         attention_mask_un[:, 0, :] = 0.  # CLS 무시
# #         sequence_output = sequence_output * attention_mask_un
# #         denom = torch.sum(attention_mask_un, dim=1, dtype=torch.float)
# #         denom = torch.clamp(denom, min=1e-6)
# #         text_out = torch.sum(sequence_output, dim=1) / denom
# #         return text_out

# #     def _eos_pooling_for_sequence(self, sequence_output, attention_mask):
# #         """
# #         sequence_output: [B, L, D]
# #         attention_mask : [B, L] (1=valid, 0=pad)
# #         → 각 배치의 마지막 유효 토큰(EOS/EOT) 히든을 골라 [B, D] 반환
# #         """
# #         lengths = attention_mask.long().sum(dim=1) - 1  # [B]
# #         L = sequence_output.size(1)
# #         lengths = torch.clamp(lengths, min=0, max=L - 1)  # OOB 방지

# #         idx = lengths.view(-1, 1, 1).expand(-1, 1, sequence_output.size(-1))  # [B,1,D]
# #         eos_vec = torch.gather(sequence_output, 1, idx).squeeze(1)            # [B,D]
# #         return eos_vec

# #     def _mean_pooling_for_similarity_visual(self, visual_output, video_mask):
# #         video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
# #         visual_output = visual_output * video_mask_un
# #         video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
# #         video_mask_un_sum[video_mask_un_sum == 0.] = 1.
# #         video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
# #         return video_out

# #     def _mean_pooling_for_similarity_audio(self, audio_output, audio_mask=None):
# #         if audio_mask is None:
# #             denom = audio_output.size(1)
# #             if denom <= 0:
# #                 denom = 1
# #             return audio_output.sum(dim=1) / float(denom)
# #         else:
# #             am = audio_mask.to(dtype=torch.float).unsqueeze(-1)
# #             audio_output = audio_output * am
# #             s = am.sum(dim=1).clamp(min=1e-6)
# #             return audio_output.sum(dim=1) / s

# #     def get_text_decomp(self, input_ids, token_type_ids, attention_mask):
# #         # 1) token-level hidden
# #         seq = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=False)  # [B,L,D]
# #         # 2) TextEditHead 적용
# #         if not hasattr(self, "text_edit_head"):
# #             self.text_edit_head = TextEditHead(dim=seq.size(-1), n_heads=1, k_sub=64).to(seq.device)
# #         out = self.text_edit_head(seq, attention_mask)
# #         # 3) L2 norm
# #         for k in out:
# #             v = out[k]
# #             v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #             out[k] = v
# #         return out  # dict


# #     # === NEW: text edit utility ===
# #     def _edit_from_ids(self, ids, seg, mask):
# #         """
# #         ids/seg/mask: [B, L]
# #         -> 편집된 g (base + γ_add*add - γ_rem*remove), L2 norm
# #         (mask가 전부 0인 배치는 자동으로 0벡터 처리)
# #         """
# #         B = ids.size(0)
# #         valid = (mask.sum(dim=1) > 0).float().unsqueeze(-1)  # [B,1]

# #         # 디폴트 하이퍼(원하면 task_config로 덮어쓰기)
# #         gamma_add = getattr(self.task_config, "gamma_add", 0.7)
# #         gamma_rem = getattr(self.task_config, "gamma_rem", 0.5)

# #         dec = self.get_text_decomp(ids, seg, mask)  # dict(base/add/remove)
# #         g = dec["base"] + gamma_add * dec["add"] - gamma_rem * dec["remove"]  # [B,D]

# #         # invalid(=빈 슬롯) 배치는 0으로
# #         g = g * valid

# #         # L2 normalize (0벡터는 안전 클램프)
# #         g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #         return g
    
# #     def get_text_global(self, input_ids, token_type_ids, attention_mask):
# #         # mean pooling 버전(기존)
# #         seq = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=False)
# #         text_global = self._mean_pooling_for_similarity_sequence(seq, attention_mask)
# #         valid = (attention_mask.sum(dim=1) > 0).float().unsqueeze(-1)
# #         text_global = text_global * valid
# #         norm = text_global.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #         text_global = (text_global / norm) * valid
# #         return text_global

# #     def get_text_global_eos(self, input_ids, token_type_ids, attention_mask):
# #         # EOS/EOT pooling 버전(새로 추가) — text_only에서 사용
# #         seq = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=False)
# #         text_global = self._eos_pooling_for_sequence(seq, attention_mask)
# #         valid = (attention_mask.sum(dim=1) > 0).float().unsqueeze(-1)
# #         text_global = text_global * valid
# #         norm = text_global.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #         text_global = (text_global / norm) * valid
# #         return text_global

# #     def _stack_text_globals_from_slots(self, ids, seg, mask, pooling="mean"):
# #         # ids/seg/mask: [B, S, L] (S<=4)
# #         B, S, _ = ids.shape
# #         outs = []
# #         valids = []
# #         for s in range(S):
# #             m = mask[:, s, :]
# #             valid = (m.sum(dim=1) > 0).float().unsqueeze(-1)
# #             if valid.any():
# #                 if pooling == "eos":
# #                     g = self.get_text_global_eos(ids[:, s, :], seg[:, s, :], m)
# #                 else:
# #                     g = self.get_text_global(ids[:, s, :], seg[:, s, :], m)
# #             else:
# #                 g = torch.zeros(B, self.hidden_size, device=ids.device, dtype=torch.float)
# #             outs.append(g)
# #             valids.append(valid)
# #         G = torch.stack(outs, dim=1)   # [B,S,D]
# #         V = torch.stack(valids, dim=1) # [B,S,1]
# #         num = V.sum(dim=1).clamp(min=1.0)  # [B,1]
# #         mean_g = (G * V).sum(dim=1) / num
# #         mean_g = mean_g / mean_g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #         return mean_g
    
# #     def _five_selfattn_mean(self, av_g, obj_g, act_g, att_g, audm_g, include_extra=True):
# #         """
# #         Self-attn over 8 tokens = [av, obj, act, att, audm, L1, L2, L3].
# #         Q=K=V=동일 시퀀스. 마지막 레이어 출력들을 (가중)평균해서 최종 q를 만듭니다.

# #         include_extra: True면 learnable 3토큰도 풀링에 포함.
# #         빈 텍스트 슬롯(전부 pad였던 경우)은 풀링에서 제외.
# #         """
# #         B, D = av_g.size()

# #         # [B,5,D] : [av | obj | act | att | audm]
# #         base = torch.stack([av_g, obj_g, act_g, att_g, audm_g], dim=1)

# #         # learnable 3 tokens -> [B,3,D]
# #         extra = self.five_sa_extra.unsqueeze(0).expand(B, -1, -1)

# #         # concat -> [B,8,D]
# #         tokens = torch.cat([base, extra], dim=1)

# #         # 마스크(풀링용): av와 extra는 항상 포함(1), 텍스트 슬롯은 유효할 때만 1
# #         txt = [obj_g, act_g, att_g, audm_g]
# #         txt_valid = [(t.norm(dim=-1) > 0).float() for t in txt]     # 각 [B]
# #         # [B,5] = [1(av), valid(obj), valid(act), valid(att), valid(audm)]
# #         pool_mask = torch.stack(
# #             [torch.ones(B, device=av_g.device, dtype=av_g.dtype)] + txt_valid,
# #             dim=1
# #         )
# #         if include_extra:
# #             extra_mask = torch.ones(B, 3, device=av_g.device, dtype=av_g.dtype)
# #         else:
# #             extra_mask = torch.zeros(B, 3, device=av_g.device, dtype=av_g.dtype)
# #         pool_mask = torch.cat([pool_mask, extra_mask], dim=1)   # [B,8]

# #         # Transformer_Text는 time-major를 받으므로 변환
# #         x = tokens.permute(1, 0, 2)  # [8,B,D]
# #         # self-attn: Q=K=V 동일, 마스크는 모두 유효
# #         q_out = self.transformer_FiveSA(x, x, None)  # 인터페이스가 (Q,KV,mask)라 None/전체유효로 호출
# #         q_out = q_out.permute(1, 0, 2).contiguous()  # [B,8,D]

# #         # (가중)평균 풀링
# #         m = pool_mask.unsqueeze(-1)                  # [B,8,1]
# #         denom = m.sum(dim=1).clamp(min=1.0)         # [B,1]
# #         pooled = (q_out * m).sum(dim=1) / denom     # [B,D]

# #         # L2 정규화
# #         pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #         return pooled


# #     # ---- low-level encoders ----
# #     def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
# #         if shaped is False:
# #             input_ids = input_ids.view(-1, input_ids.shape[-1])
# #             token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
# #             attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
# #         bs_pair = input_ids.size(0)
# #         if input_ids.shape[1] != 20:
# #             sequence_hidden = self.clip.encode_text(input_ids).float()          # expect [B, L, D]
# #             sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))
# #         else:
# #             sequence_hidden = [self.clip.encode_text(input_ids[:, i, :]).float() for i in range(input_ids.shape[1])]
# #             sequence_hidden = torch.stack(sequence_hidden, dim=1)
# #         return sequence_hidden

# #     def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
# #         if shaped is False:
# #             video_mask = video_mask.view(-1, video_mask.shape[-1])
# #             video = torch.as_tensor(video).float()
# #             b, pair, bs, ts, channel, h, w = video.shape
# #             video = video.view(b * pair * bs * ts, channel, h, w)
# #             video_frame = bs * ts
# #         bs_pair = video_mask.size(0)
# #         visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
# #         visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
# #         return visual_hidden

# #     def get_audio_output(self, audio):
# #         audio = audio.squeeze(1)  # (B,1,T,128) -> (B,T,128)
# #         audio_hidden = self.clip.encode_audio(audio).float()
# #         return audio_hidden



# #     # ================================ #
# #     # === [ADDED] proj 분해 유틸  ===== #
# #     # ================================ #
# #     def _disent_one_slot_proj(self, av_g, t_g):
# #         """
# #         av_g, t_g: [B, D]
# #         반환: rel, unr, mask  (각 [B,D], [B,D], [B])
# #         """
# #         eps = 1e-6
# #         av = av_g / av_g.norm(dim=-1, keepdim=True).clamp(min=eps)

# #         t_norm = t_g.norm(dim=-1, keepdim=True)
# #         mask = (t_norm > 1e-6).squeeze(-1)           # [B]
# #         if not mask.any():
# #             B, D = av_g.shape
# #             z = torch.zeros(B, D, device=av_g.device, dtype=av_g.dtype)
# #             return z, z, mask

# #         t_unit = torch.where(t_norm > 1e-6, t_g / t_norm, torch.zeros_like(t_g))

# #         coeff = (av * t_unit).sum(dim=-1, keepdim=True)   # [B,1]
# #         rel = coeff * t_unit                              # [B,D]
# #         res = av - rel                                    # [B,D]

# #         rel = rel / rel.norm(dim=-1, keepdim=True).clamp(min=eps)
# #         resn = res.norm(dim=-1, keepdim=True)
# #         unr = torch.where(resn > eps, res / resn, torch.zeros_like(res))

# #         rel = rel * mask.unsqueeze(-1)
# #         unr = unr * mask.unsqueeze(-1)
# #         return rel, unr, mask

# #     def _gate_per_slot_simple(self, av_g, rel, unr):
# #         x = torch.cat([av_g, rel, unr], dim=-1)  # [B, 3D]
# #         w = self.disent_gate_simple(x)           # [B,2], sigmoid
# #         return w[:, :1], w[:, 1:]
# #     # ================================ #
# #     # === [/ADDED] proj 분해 유틸 ==== #
# #     # ================================ #


# #     # ---- Public APIs ----
# #     def encode_query(self, q_input_ids, q_token_type_ids, q_attention_mask, q_video, q_video_mask, q_audio):
# #         # std video mask
# #         if q_video_mask.dim() == 3 and q_video_mask.size(1) == 1:
# #             q_video_mask = q_video_mask.squeeze(1)
# #         elif q_video_mask.dim() != 2:
# #             q_video_mask = q_video_mask.view(q_video_mask.size(0), -1)

# #         rmode = self.retrieval_mode

# #         # ===== 비텍스트 기반 질의 모드 =====
# #         if rmode == "video_only":
# #             v = self.get_visual_output(q_video, q_video_mask, shaped=False)
# #             v = self._mean_pooling_for_similarity_visual(v, q_video_mask)
# #             return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)

# #         if rmode == "audio_only":
# #             a = self.get_audio_output(q_audio)                   # [B,Ta,D]
# #             # a = self._mean_pooling_for_similarity_audio(a, None) # no mask
# #             return self._global_from_audio(a, how=getattr(self.task_config, "audio_pooling", "cls"))

# #         if rmode == "text_only":
# #             if q_input_ids.dim() == 3:
# #                 # 4 슬롯 각각 EOS로 뽑고 → 평균
# #                 t = self._stack_text_globals_from_slots(q_input_ids, q_token_type_ids, q_attention_mask, pooling="eos")
# #                 return t
# #             else:
# #                 # 단일 텍스트도 EOS로
# #                 t = self.get_text_global_eos(q_input_ids, q_token_type_ids, q_attention_mask)
# #                 return t

# #         if rmode == "va_only":
# #             v = self.get_visual_output(q_video, q_video_mask, shaped=False)
# #             a = self.get_audio_output(q_audio)
# #             va, _, _ = self._fuse_av_only(v, q_video_mask, a)
# #             return va / va.norm(dim=-1, keepdim=True).clamp(min=1e-6)

# #         # ===== 텍스트 기반 전략 =====
# #         if rmode == "concat_sum":
# #             if q_input_ids.dim() == 3:
# #                 q_ids = q_input_ids[:, 0, :]
# #                 q_seg = q_token_type_ids[:, 0, :]
# #                 q_mask = q_attention_mask[:, 0, :]
# #             else:
# #                 q_ids, q_seg, q_mask = q_input_ids, q_token_type_ids, q_attention_mask
# #             seq = self.get_sequence_output(q_ids, q_seg, q_mask, shaped=False)
# #             t = self._mean_pooling_for_similarity_sequence(seq, q_mask)
# #             t = t / t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #             v = self.get_visual_output(q_video, q_video_mask, shaped=False)
# #             a = self.get_audio_output(q_audio)
# #             va, _, _ = self._fuse_av_only(v, q_video_mask, a)
# #             q = t + va
# #             return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)

# #         if rmode == "five_sum":
# #             # 4개 슬롯 체크
# #             if q_input_ids.dim() != 3 or q_input_ids.size(1) < 4:
# #                 raise ValueError(f"five_sum requires [B,4,L] input shape")
            
# #             # 4개 텍스트 각각 인코딩
# #             obj_g = self.get_text_global(
# #                 q_input_ids[:, 0, :], 
# #                 q_token_type_ids[:, 0, :], 
# #                 q_attention_mask[:, 0, :]
# #             )
# #             act_g = self.get_text_global(
# #                 q_input_ids[:, 1, :], 
# #                 q_token_type_ids[:, 1, :], 
# #                 q_attention_mask[:, 1, :]
# #             )
# #             att_g = self.get_text_global(
# #                 q_input_ids[:, 2, :], 
# #                 q_token_type_ids[:, 2, :], 
# #                 q_attention_mask[:, 2, :]
# #             )
# #             audm_g = self.get_text_global(
# #                 q_input_ids[:, 3, :], 
# #                 q_token_type_ids[:, 3, :], 
# #                 q_attention_mask[:, 3, :]
# #             )
            
# #             # AV 융합
# #             v = self.get_visual_output(q_video, q_video_mask, shaped=False)
# #             a = self.get_audio_output(q_audio)
# #             av, _, _ = self._fuse_av_only(v, q_video_mask, a)
            
            
# #             q = av + (obj_g + act_g + att_g + audm_g) * 0.25
            
# #             # 정규화
# #             return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        
# #         if rmode == "textquery_fuse":
# #             # text-queries = obj/act/att ; KV = audm (or real audio)
# #             q_list = []
# #             for g in (obj_g, act_g, att_g):
# #                 valid = (g.norm(dim=-1, keepdim=True) > 0).float()
# #                 q_list.append(g * valid)
# #             text_queries = torch.stack(q_list, dim=1)  # [B,3,D]

# #             use_audm = (audm_g.norm(dim=-1) > 0).any().item()
# #             if use_audm:
# #                 kv_tokens = audm_g.unsqueeze(1)                     # [B,1,D]
# #             else:
# #                 audio_tokens = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #                 kv_tokens = audio_tokens                            # [B,Ta,D]

# #             q_va, _, _ = self._fuse_av_textquery(v, q_video_mask, text_queries, kv_tokens)
# #             return q_va / q_va.norm(dim=-1, keepdim=True).clamp(min=1e-6)

# #         # [encode_query] 내부, five_sum 바로 아래에 추가

# #         if rmode == "five_weighted":
# #             if q_input_ids.dim() != 3 or q_input_ids.size(1) < 4:
# #                 raise ValueError(f"five_weighted requires [B,4,L] input shape")

# #             obj_g = self.get_text_global(q_input_ids[:, 0, :], q_token_type_ids[:, 0, :], q_attention_mask[:, 0, :])  # [B,D]
# #             act_g = self.get_text_global(q_input_ids[:, 1, :], q_token_type_ids[:, 1, :], q_attention_mask[:, 1, :])
# #             att_g = self.get_text_global(q_input_ids[:, 2, :], q_token_type_ids[:, 2, :], q_attention_mask[:, 2, :])
# #             audm_g = self.get_text_global(q_input_ids[:, 3, :], q_token_type_ids[:, 3, :], q_attention_mask[:, 3, :])

# #             # AV
# #             v = self.get_visual_output(q_video, q_video_mask, shaped=False)
# #             a = self.get_audio_output(q_audio)
# #             av_g, _, _ = self._fuse_av_only(v, q_video_mask, a)   # [B,D] (정규화됨)

# #             q, w = self._five_weighted(av_g, obj_g, act_g, att_g, audm_g)

# #             self._cached_five_weights = w
# #             return q
        
# #         if rmode == "five_weighted_without_obj" or rmode == "five_weighted_without_act" or rmode == "five_weighted_without_att" or rmode == "five_weighted_without_audm":
# #             if q_input_ids.dim() != 3 or q_input_ids.size(1) < 4:
# #                 raise ValueError(f"{rmode} requires [B,4,L] input shape")

# #             obj_g = self.get_text_global(q_input_ids[:, 0, :], q_token_type_ids[:, 0, :], q_attention_mask[:, 0, :])  # [B,D]
# #             act_g = self.get_text_global(q_input_ids[:, 1, :], q_token_type_ids[:, 1, :], q_attention_mask[:, 1, :])
# #             att_g = self.get_text_global(q_input_ids[:, 2, :], q_token_type_ids[:, 2, :], q_attention_mask[:, 2, :])
# #             audm_g = self.get_text_global(q_input_ids[:, 3, :], q_token_type_ids[:, 3, :], q_attention_mask[:, 3, :])

# #             # AV
# #             v = self.get_visual_output(q_video, q_video_mask, shaped=False)
# #             a = self.get_audio_output(q_audio)
# #             av_g, _, _ = self._fuse_av_only(v, q_video_mask, a)   # [B,D] (정규화됨)

# #             if rmode == "five_weighted_without_obj":
# #                 q, w = self._five_weighted_without_one(av_g, act_g, att_g, audm_g)
# #             elif rmode == "five_weighted_without_act":
# #                 q, w = self._five_weighted_without_one(av_g, obj_g, att_g, audm_g)
# #             elif rmode == "five_weighted_without_att":
# #                 q, w = self._five_weighted_without_one(av_g, obj_g, act_g, audm_g)
# #             else: # without_audm
# #                 q, w = self._five_weighted_without_one(av_g, obj_g, act_g, att_g)

# #             self._cached_five_weights = w
# #             return q

# #         if rmode == "five_selfattn_mean":
# #             if q_input_ids.dim() != 3 or q_input_ids.size(1) < 4:
# #                 raise ValueError("five_selfattn_mean requires [B,4,L] input shape")

# #             # 텍스트 글로벌 (기존과 동일한 방식 사용; 필요 시 eos 버전으로 바꿔도 됨)
# #             obj_g = self.get_text_global(q_input_ids[:, 0, :], q_token_type_ids[:, 0, :], q_attention_mask[:, 0, :])
# #             act_g = self.get_text_global(q_input_ids[:, 1, :], q_token_type_ids[:, 1, :], q_attention_mask[:, 1, :])
# #             att_g = self.get_text_global(q_input_ids[:, 2, :], q_token_type_ids[:, 2, :], q_attention_mask[:, 2, :])
# #             audm_g = self.get_text_global(q_input_ids[:, 3, :], q_token_type_ids[:, 3, :], q_attention_mask[:, 3, :])

# #             # AV 글로벌
# #             v = self.get_visual_output(q_video, q_video_mask, shaped=False)
# #             a = self.get_audio_output(q_audio)
# #             av_g, _, _ = self._fuse_av_only(v, q_video_mask, a)  # [B,D]

# #             q = self._five_selfattn_mean(av_g, obj_g, act_g, att_g, audm_g,
# #                                         include_extra=self.five_sa_pool_include_extra)
# #             self._cached_five_weights = None  # 이 경로는 스칼라 게이트가 없으므로 None
# #             return q


# #         if rmode == "five_weighted_independent":
# #             if q_input_ids.dim() != 3 or q_input_ids.size(1) < 4:
# #                 raise ValueError("five_weighted_independent requires [B,4,L] input shape")

# #             # 4 text globals (기존 함수 재사용)
# #             obj_g = self.get_text_global(q_input_ids[:, 0, :], q_token_type_ids[:, 0, :], q_attention_mask[:, 0, :])
# #             act_g = self.get_text_global(q_input_ids[:, 1, :], q_token_type_ids[:, 1, :], q_attention_mask[:, 1, :])
# #             att_g = self.get_text_global(q_input_ids[:, 2, :], q_token_type_ids[:, 2, :], q_attention_mask[:, 2, :])
# #             audm_g = self.get_text_global(q_input_ids[:, 3, :], q_token_type_ids[:, 3, :], q_attention_mask[:, 3, :])

# #             # AV global
# #             v = self.get_visual_output(q_video, q_video_mask, shaped=False)
# #             a = self.get_audio_output(q_audio)
# #             av_g, _, _ = self._fuse_av_only(v, q_video_mask, a)  # [B, D]

# #             q, weights = self._five_weighted_independent(av_g, obj_g, act_g, att_g, audm_g)
# #             self._cached_five_weights = weights  # {"pair":[B,4,2], "mix":[B,4]}
# #             return q


# #         if rmode == "five_crossattn":
# #             if q_input_ids.dim() != 3 or q_input_ids.size(1) < 4:
# #                 raise ValueError(f"five_crossattn requires [B,4,L] input shape")

# #             # 4개 텍스트 슬롯 글로벌 벡터
# #             obj_g = self.get_text_global(q_input_ids[:, 0, :], q_token_type_ids[:, 0, :], q_attention_mask[:, 0, :])  # [B,D]
# #             act_g = self.get_text_global(q_input_ids[:, 1, :], q_token_type_ids[:, 1, :], q_attention_mask[:, 1, :])
# #             att_g = self.get_text_global(q_input_ids[:, 2, :], q_token_type_ids[:, 2, :], q_attention_mask[:, 2, :])
# #             audm_g = self.get_text_global(q_input_ids[:, 3, :], q_token_type_ids[:, 3, :], q_attention_mask[:, 3, :])

# #             # AV 융합 글로벌(쿼리로 사용)
# #             v = self.get_visual_output(q_video, q_video_mask, shaped=False)
# #             a = self.get_audio_output(q_audio)
# #             av_g, _, _ = self._fuse_av_only(v, q_video_mask, a)  # [B,D]

# #             # 크로스 어텐션으로 최종 q 생성
# #             q = self._five_crossattn(av_g, obj_g, act_g, att_g, audm_g)
# #             # (cross-attn 경로는 weight 스칼라가 없으므로 캐시 초기화)
# #             self._cached_five_weights = None
# #             return q
        
# #         if rmode == "slerp_base":
# #             if q_input_ids.dim() != 3 or q_input_ids.size(1) < 4:
# #                 raise ValueError("slerp_base requires [B,4,L]")

# #             v = self.get_visual_output(q_video, q_video_mask, shaped=False)
# #             a = self.get_audio_output(q_audio)
# #             av_g, _, _ = self._fuse_av_only(v, q_video_mask, a)

# #             obj_g = self.get_text_global(q_input_ids[:,0,:], q_token_type_ids[:,0,:], q_attention_mask[:,0,:])
# #             act_g = self.get_text_global(q_input_ids[:,1,:], q_token_type_ids[:,1,:], q_attention_mask[:,1,:])
# #             att_g = self.get_text_global(q_input_ids[:,2,:], q_token_type_ids[:,2,:], q_attention_mask[:,2,:])
# #             audm_g= self.get_text_global(q_input_ids[:,3,:], q_token_type_ids[:,3,:], q_attention_mask[:,3,:])

# #             B = av_g.size(0); one = torch.ones(B,1, device=av_g.device, dtype=av_g.dtype)
# #             alpha_obj = 0.5 * one; alpha_act = 0.5 * one; alpha_att = 0.5 * one; alpha_audm = 0.5 * one

# #             av_obj  = self._slerp(av_g, obj_g,  alpha_obj)
# #             av_act  = self._slerp(av_g, act_g,  alpha_act)
# #             av_att  = self._slerp(av_g, att_g,  alpha_att)
# #             av_audm = self._slerp(av_g, audm_g, alpha_audm)

# #             specialized = torch.stack([av_obj, av_act, av_att, av_audm], dim=1)  # [B,4,D]
# #             weights = torch.tensor([1.0,1.0,0.5,1.0], device=av_g.device, dtype=av_g.dtype)
# #             weights = weights.unsqueeze(0).expand(B, -1)

# #             q_final = self._spherical_mean(specialized, weights=weights)
# #             return q_final

# #         if rmode == "av_disentangle_slots_proj":
# #             # AV 글로벌
# #             v = self.get_visual_output(q_video, q_video_mask, shaped=False)
# #             a = self.get_audio_output(q_audio)
# #             av_g, _, _ = self._fuse_av_only(v, q_video_mask, a)           # [B,D]

# #             # 슬롯 텍스트 4개 글로벌 (obj/act/att/audm)
# #             T = []
# #             for s in range(4):
# #                 T.append(self.get_text_global_eos(
# #                     q_input_ids[:, s, :], q_token_type_ids[:, s, :], q_attention_mask[:, s, :]
# #                 ))  # [B,D]

# #             # 슬롯별 분해 & 게이트 & 합성
# #             q_sum = av_g.clone()
# #             rel_list = []
# #             unr_list = []
# #             mask_list = []

# #             for t_i in T:
# #                 rel_i, unr_i, m_i = self._disent_one_slot_proj(av_g, t_i)     # proj 분해
# #                 w1_i, w2_i = self._gate_per_slot_simple(av_g, rel_i, unr_i)   # 단순 게이트
# #                 q_sum = q_sum + m_i.unsqueeze(-1) * (w1_i * rel_i + w2_i * unr_i)
# #                 rel_list.append(rel_i); unr_list.append(unr_i); mask_list.append(m_i)

# #             q = q_sum / q_sum.norm(dim=-1, keepdim=True).clamp(min=1e-6)

# #             # 손실용 캐시 (직교 손실에만 사용)
# #             self._cached_rel = torch.stack(rel_list, dim=1)    # [B,4,D]
# #             self._cached_unr = torch.stack(unr_list, dim=1)    # [B,4,D]
# #             self._cached_mask = torch.stack(mask_list, dim=1)  # [B,4]
# #             return q
        
# #         # encode_query(...), rmode 분기들 중 slerp_base 아래(또는 그 자리에 교체)로 추가

# #         if rmode == "slerp1":
# #             # 필요 텍스트 슬롯 확인
# #             if q_input_ids.dim() != 3 or q_input_ids.size(1) < 4:
# #                 raise ValueError("slerp1 requires [B,4,L]")

# #             # AV 글로벌
# #             v = self.get_visual_output(q_video, q_video_mask, shaped=False)
# #             a = self.get_audio_output(q_audio)
# #             av_g, _, _ = self._fuse_av_only(v, q_video_mask, a)     # [B,D]

# #             # 4개 텍스트 글로벌
# #             obj_g = self.get_text_global_eos(q_input_ids[:,0,:], q_token_type_ids[:,0,:], q_attention_mask[:,0,:])
# #             act_g = self.get_text_global_eos(q_input_ids[:,1,:], q_token_type_ids[:,1,:], q_attention_mask[:,1,:])
# #             att_g = self.get_text_global_eos(q_input_ids[:,2,:], q_token_type_ids[:,2,:], q_attention_mask[:,2,:])
# #             audm_g= self.get_text_global_eos(q_input_ids[:,3,:], q_token_type_ids[:,3,:], q_attention_mask[:,3,:])

# #             # 유효 슬롯만 평균 (마스크 고려)
# #             T = torch.stack([obj_g, act_g, att_g, audm_g], dim=1)          # [B,4,D]
# #             mask = (T.norm(dim=-1) > 0).float()                            # [B,4]
# #             denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)           # [B,1]
# #             T_mean = (T * mask.unsqueeze(-1)).sum(dim=1) / denom           # [B,D]
# #             T_mean = T_mean / T_mean.norm(dim=-1, keepdim=True).clamp(min=1e-6)

# #             # tau in [0,1]
# #             tau = torch.sigmoid(self.slerp1_tau)
# #             tau = tau.expand(av_g.size(0), 1)                              # [B,1]

# #             q_final = self._slerp(av_g, T_mean, tau)                       # [B,D]
# #             return q_final

# #         if rmode == "slerp2":
# #             if q_input_ids.dim() != 3 or q_input_ids.size(1) < 4:
# #                 raise ValueError("slerp2 requires [B,4,L]")

# #             # AV 글로벌
# #             v = self.get_visual_output(q_video, q_video_mask, shaped=False)
# #             a = self.get_audio_output(q_audio)
# #             av_g, _, _ = self._fuse_av_only(v, q_video_mask, a)            # [B,D]

# #             # 4개 텍스트 글로벌
# #             obj_g = self.get_text_global_eos(q_input_ids[:,0,:], q_token_type_ids[:,0,:], q_attention_mask[:,0,:])
# #             act_g = self.get_text_global_eos(q_input_ids[:,1,:], q_token_type_ids[:,1,:], q_attention_mask[:,1,:])
# #             att_g = self.get_text_global_eos(q_input_ids[:,2,:], q_token_type_ids[:,2,:], q_attention_mask[:,2,:])
# #             audm_g= self.get_text_global_eos(q_input_ids[:,3,:], q_token_type_ids[:,3,:], q_attention_mask[:,3,:])

# #             T = [obj_g, act_g, att_g, audm_g]

# #             # (1) AV–각 슬롯 slerp (tau는 학습 파라미터 4개)
# #             tau = torch.sigmoid(self.slerp2_tau)                           # [4]
# #             B = av_g.size(0)
# #             slerped = []
# #             for i in range(4):
# #                 t_i = tau[i].expand(B, 1)                                  # [B,1]
# #                 s_i = self._slerp(av_g, T[i], t_i)                         # [B,D]
# #                 slerped.append(s_i)
# #             S = torch.stack(slerped, dim=1)                                # [B,4,D]

# #             # (2) 4개 결과의 구면 평균(가중치는 softmax로 학습)
# #             w = torch.softmax(self.slerp2_weight_logits, dim=-1)           # [4]
# #             w = w.unsqueeze(0).expand(B, -1)                               # [B,4]
# #             q_final = self._spherical_mean(S, weights=w)                   # [B,D]
# #             return q_final

# #         if rmode == "five_pairattn_mean":
# #             if q_input_ids.dim() != 3 or q_input_ids.size(1) < 4:
# #                 raise ValueError("five_pairattn_mean requires [B,4,L] input shape")

# #             # 4 text globals (mean or eos 중 원하는 걸로; 여기선 기존 mean 버전 사용)
# #             obj_g = self.get_text_global(q_input_ids[:, 0, :], q_token_type_ids[:, 0, :], q_attention_mask[:, 0, :])
# #             act_g = self.get_text_global(q_input_ids[:, 1, :], q_token_type_ids[:, 1, :], q_attention_mask[:, 1, :])
# #             att_g = self.get_text_global(q_input_ids[:, 2, :], q_token_type_ids[:, 2, :], q_attention_mask[:, 2, :])
# #             audm_g = self.get_text_global(q_input_ids[:, 3, :], q_token_type_ids[:, 3, :], q_attention_mask[:, 3, :])

# #             # AV global
# #             v = self.get_visual_output(q_video, q_video_mask, shaped=False)
# #             a = self.get_audio_output(q_audio)
# #             av_g, _, _ = self._fuse_av_only(v, q_video_mask, a)  # [B, D]

# #             q, weights = self._five_pairattn_mean(
# #                 av_g, obj_g, act_g, att_g, audm_g,
# #                 include_extra=self.five_pairattn_pool_include_extra
# #             )
# #             self._cached_five_weights = weights  # {"pair":[B,4,2]}
# #             return q

# #         # ====== 간단 AVG 버전: V+T vs V+A ======
# #         if rmode == "vt_only":
# #             # V 글로벌
# #             v_tok = self.get_visual_output(q_video, q_video_mask, shaped=False)          # [B, Tv, D]
# #             v_g   = self._mean_pooling_for_similarity_visual(v_tok, q_video_mask)        # [B, D]
# #             v_g   = v_g / v_g.norm(dim=-1, keepdim=True).clamp(min=1e-6)

# #             # T 글로벌 (슬롯 입력이면 슬롯 평균, 단일 입력이면 그대로)
# #             if q_input_ids.dim() == 3:
# #                 # [B, S(≤4), L] -> 슬롯별 글로벌 평균 후 valid만 평균
# #                 t_g = self._stack_text_globals_from_slots(q_input_ids, q_token_type_ids, q_attention_mask, pooling="mean")
# #             else:
# #                 t_g = self.get_text_global(q_input_ids, q_token_type_ids, q_attention_mask)  # [B, D]

# #             # AVG → 정규화
# #             q = 0.5 * (v_g + t_g)
# #             q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #             return q

# #         # ====== 간단 AVG 버전: A+T vs V+A ======
# #         if rmode == "at_only":
# #             # A 글로벌
# #             a_tok = self.get_audio_output(q_audio)                                       # [B, Ta, D]
# #             a_g   = self._global_from_audio(a_tok, how=getattr(self.task_config, "audio_pooling", "cls"))  # [B, D]

# #             # T 글로벌
# #             if q_input_ids.dim() == 3:
# #                 t_g = self._stack_text_globals_from_slots(q_input_ids, q_token_type_ids, q_attention_mask, pooling="mean")
# #             else:
# #                 t_g = self.get_text_global(q_input_ids, q_token_type_ids, q_attention_mask)  # [B, D]

# #             # AVG → 정규화
# #             q = 0.5 * (a_g + t_g)
# #             q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #             return q



        
# #         raise ValueError(f"Unknown retrieval_mode: {rmode}")

# #     def encode_target(self, t_video, t_video_mask, t_audio):
# #         if t_video_mask.dim() == 3 and t_video_mask.size(1) == 1:
# #             t_video_mask = t_video_mask.squeeze(1)
# #         elif t_video_mask.dim() != 2:
# #             t_video_mask = t_video_mask.view(t_video_mask.size(0), -1)

# #         t_visual_output = self.get_visual_output(t_video, t_video_mask, shaped=False)
# #         t_audio_output = self.get_audio_output(t_audio)
# #         t_av_global, _, _ = self._fuse_av_only(t_visual_output, t_video_mask, t_audio_output)
# #         return t_av_global

# #     def _global_from_audio(self, a, how="cls"):
# #         """
# #         a: [B, Ta, D]  (encode_audio 출력)
# #         how: "cls" or "mean"
# #         """
# #         if how == "cls":
# #             g = a[:, 0, :]           # 0번째 토큰 = (cls+dist)/2 after AST & projection
# #         else:
# #             g = a.mean(dim=1)
# #         return g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)

# #     # ==== fusion blocks ====
# #     def _fuse_av_only(self, visual_output, video_mask, audio_output):
# #         query_tokens = self.query_tokens.t().unsqueeze(0)
# #         query_embed = query_tokens.expand(visual_output.shape[0], -1, -1)

# #         position_ids = torch.arange(query_embed.size(1), dtype=torch.long, device=visual_output.device)
# #         position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
# #         query_position_embeddings = self.query_position_embeddings(position_ids)
# #         query_embed = query_embed + query_position_embeddings

# #         query_mask = torch.ones(visual_output.shape[0], query_embed.shape[1], device=visual_output.device)
# #         extended_query_mask = (1.0 - query_mask.unsqueeze(-1)) * -1000000.0
# #         extended_query_mask = extended_query_mask.expand(-1, -1, audio_output.size(1))

# #         query_embed = query_embed.permute(1, 0, 2)
# #         audio_output = audio_output.permute(1, 0, 2)
# #         qa_output = self.transformer_Fusion(query_embed, audio_output, extended_query_mask)
# #         qa_output = qa_output.permute(1, 0, 2).contiguous()
# #         audio_output = audio_output.permute(1, 0, 2).contiguous()

# #         seq_length = visual_output.size(1)
# #         position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
# #         position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
# #         frame_position_embeddings = self.frame_position_embeddings(position_ids)

# #         visual_output_original = visual_output
# #         visual_output = visual_output + frame_position_embeddings

# #         extended_video_mask = (1.0 - video_mask.unsqueeze(-1)) * -1000000.0
# #         extended_video_mask = extended_video_mask.expand(-1, -1, qa_output.size(1))

# #         qa_output = qa_output.permute(1, 0, 2)
# #         visual_output = visual_output.permute(1, 0, 2)
# #         fusion_output, _, _, attn_gate_list, ff_gate_list = self.transformerClip(visual_output, qa_output, extended_video_mask)
# #         fusion_output = fusion_output.permute(1, 0, 2)

# #         fusion_output = 0.05 * fusion_output + 0.95 * visual_output_original
# #         fusion_output = fusion_output.contiguous()

# #         av_fused_output_global = self._mean_pooling_for_similarity_visual(fusion_output, video_mask)
# #         av_fused_output_global = av_fused_output_global / av_fused_output_global.norm(dim=-1, keepdim=True)

# #         attn_gate_tensor = torch.stack(attn_gate_list, 1) if isinstance(attn_gate_list, (list, tuple)) else None
# #         ff_gate_tensor = torch.stack(ff_gate_list, 1) if isinstance(ff_gate_list, (list, tuple)) else None
# #         return av_fused_output_global, attn_gate_tensor, ff_gate_tensor

# #     def _fuse_av_textquery(self, visual_output, video_mask, text_queries, kv_tokens):
# #         N, T, D = visual_output.shape
# #         Q = text_queries.size(1)

# #         position_ids = torch.arange(Q, dtype=torch.long, device=visual_output.device)
# #         position_ids = position_ids.unsqueeze(0).expand(N, -1)
# #         query_position_embeddings = self.query_position_embeddings(position_ids)
# #         query_embed = text_queries + query_position_embeddings

# #         query_mask = torch.ones(N, Q, device=visual_output.device)
# #         extended_query_mask = (1.0 - query_mask.unsqueeze(-1)) * -1000000.0
# #         extended_query_mask = extended_query_mask.expand(-1, -1, kv_tokens.size(1))

# #         query_embed = query_embed.permute(1, 0, 2)
# #         kv_tokens = kv_tokens.permute(1, 0, 2)
# #         qa_output = self.transformer_Fusion(query_embed, kv_tokens, extended_query_mask)
# #         qa_output = qa_output.permute(1, 0, 2).contiguous()
# #         kv_tokens = kv_tokens.permute(1, 0, 2).contiguous()

# #         position_ids = torch.arange(T, dtype=torch.long, device=visual_output.device)
# #         position_ids = position_ids.unsqueeze(0).expand(N, -1)
# #         frame_position_embeddings = self.frame_position_embeddings(position_ids)

# #         visual_output_original = visual_output
# #         visual_output = visual_output + frame_position_embeddings

# #         extended_video_mask = (1.0 - video_mask.unsqueeze(-1)) * -1000000.0
# #         extended_video_mask = extended_video_mask.expand(-1, -1, Q)

# #         qa_output = qa_output.permute(1, 0, 2)
# #         visual_output = visual_output.permute(1, 0, 2)
# #         fusion_output, _, _, attn_gate_list, ff_gate_list = self.transformerClip(visual_output, qa_output, extended_video_mask)
# #         fusion_output = fusion_output.permute(1, 0, 2)

# #         fusion_output = 0.05 * fusion_output + 0.95 * visual_output_original
# #         fusion_output = fusion_output.contiguous()

# #         av_fused_output_global = self._mean_pooling_for_similarity_visual(fusion_output, video_mask)
# #         av_fused_output_global = av_fused_output_global / av_fused_output_global.norm(dim=-1, keepdim=True)

# #         attn_gate_tensor = torch.stack(attn_gate_list, 1) if isinstance(attn_gate_list, (list, tuple)) else None
# #         ff_gate_tensor = torch.stack(ff_gate_list, 1) if isinstance(ff_gate_list, (list, tuple)) else None
# #         return av_fused_output_global, attn_gate_tensor, ff_gate_tensor


# #     def _slerp(self, v0, v1, t, DOT_THRESHOLD=0.9995):
# #         v0 = v0 / v0.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #         # v1 무효(거의 영벡터)면 그냥 v0 반환
# #         v1_norm = v1.norm(dim=-1, keepdim=True)
# #         v1 = torch.where(v1_norm > 1e-6, v1 / v1_norm, v0)

# #         dot = torch.sum(v0 * v1, dim=-1, keepdim=True).clamp(-1.0, 1.0)
# #         mask = (torch.abs(dot) > DOT_THRESHOLD).squeeze(-1)

# #         theta = torch.acos(dot); sin_theta = torch.sin(theta)
# #         sin_theta_safe = torch.where(sin_theta == 0, torch.ones_like(sin_theta), sin_theta)
# #         s0 = torch.sin((1.0 - t) * theta) / sin_theta_safe
# #         s1 = torch.sin(t * theta) / sin_theta_safe
# #         v2 = s0 * v0 + s1 * v1

# #         lerp_res = v0 + t * (v1 - v0)
# #         lerp_res = lerp_res / lerp_res.norm(dim=-1, keepdim=True).clamp(min=1e-6)

# #         res = torch.where(mask.unsqueeze(-1), lerp_res, v2)
# #         # 최종 정규화
# #         res = res / res.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #         return res

# #     def _spherical_mean(self, vectors, weights=None, iterations=4):
# #         """
# #         주어진 벡터들의 (가중) 구면 평균을 계산합니다.
# #         vectors: [B, N, D] (N은 벡터의 개수, 여기서는 4)
# #         weights: [B, N] (선택 사항, 가중치)
# #         """
# #         B, N, D = vectors.shape
        
# #         # 가중치가 없으면 동일한 가중치(1/N)를 적용
# #         if weights is None:
# #             weights = torch.ones(B, N, device=vectors.device) / N
# #         else:
# #             # 가중치의 합이 1이 되도록 정규화
# #             weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        
# #         weights = weights.unsqueeze(-1) # [B, N, 1]

# #         # 초기 추정치로 유클리드 평균 사용
# #         mean_vec = torch.sum(vectors * weights, dim=1)
# #         mean_vec = mean_vec / mean_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)

# #         # 반복적으로 평균을 업데이트
# #         for _ in range(iterations):
# #             # 현재 평균 벡터를 기준으로 가중 유클리드 평균 계산
# #             mean_vec_weighted_sum = torch.sum(vectors * weights, dim=1)
# #             # 새로운 평균으로 투영 (정규화)
# #             mean_vec = mean_vec_weighted_sum / mean_vec_weighted_sum.norm(dim=-1, keepdim=True).clamp(min=1e-6)

# #         return mean_vec

# #     def forward(self,
# #                 q_input_ids, q_token_type_ids, q_attention_mask,
# #                 q_video, q_video_mask, q_audio,
# #                 t_video, t_video_mask, t_audio):
# #         assert self.sim_header == "seqTransf", "Gated Fusion(Transformer) required."

# #         q_global = self.encode_query(q_input_ids, q_token_type_ids, q_attention_mask, q_video, q_video_mask, q_audio)
# #         t_av_global = self.encode_target(t_video, t_video_mask, t_audio)
# #         logit_scale = self.clip.logit_scale.exp()

# #         if self.training:
# #             q_all = allgather(q_global, self.task_config)
# #             t_all = allgather(t_av_global, self.task_config)
# #             torch.distributed.barrier()
# #             sim_matrix = torch.matmul(q_all, t_all.t()) * logit_scale
# #             loss = (self.loss_fct_vis(sim_matrix) + self.loss_fct_vis(sim_matrix.t())) / 2.0
# #             return loss
# #         else:
# #             sim_matrix = torch.matmul(q_global, t_av_global.t()) * logit_scale
# #             return sim_matrix, (None, None, None, None)
        
# # # modules/modeling.py 맨 위 import 근처
# # from torch.nn.utils import parametrize
# # from torch.nn.utils import parametrize

# # class _OrthoProjector(nn.Module):
# #     """x -> U U^T x, U: (D, k), 열 직교를 보장 (Parametrization)"""
# #     def __init__(self, dim, k):
# #         super().__init__()
# #         W = torch.empty(dim, k)
# #         nn.init.orthogonal_(W)
# #         self.W = nn.Parameter(W)  # (D,k)
# #         parametrize.register_parametrization(self, "W", _ColumnOrth())  # 아래 참고

# #     def forward(self, x):  # x: (..., D)
# #         U = self.W  # 직교화된 열벡터
# #         return (x @ U) @ U.t()

# # class _ColumnOrth(nn.Module):
# #     def forward(self, X):
# #         # Y = qr(X) 의 Q만 사용 (열직교)
# #         # 주: 배치 없는 작은 행렬이라 비용 미미
# #         Q, _ = torch.linalg.qr(X, mode='reduced')
# #         return Q

# # class TextEditHead(nn.Module):
# #     """
# #     입력: token-level hidden [B, L, D], mask [B, L]
# #     출력: dict(base/add/remove) 각 [B, D]
# #     """
# #     def __init__(self, dim, n_heads=1, k_sub=64):
# #         super().__init__()
# #         self.q_base   = nn.Parameter(torch.randn(n_heads, dim))
# #         self.q_add    = nn.Parameter(torch.randn(n_heads, dim))
# #         self.q_remove = nn.Parameter(torch.randn(n_heads, dim))
# #         nn.init.normal_(self.q_base,   std=0.02)
# #         nn.init.normal_(self.q_add,    std=0.02)
# #         nn.init.normal_(self.q_remove, std=0.02)

# #         # 토큰 어텐션 → 요약
# #         self.proj = nn.Linear(dim, dim)  # 가벼운 토큰 프로젝션
# #         # 서브스페이스 투영(직교화): base/add/remove가 서로 엇갈리게
# #         self.P_add = _OrthoProjector(dim, k_sub)
# #         self.P_rem = _OrthoProjector(dim, k_sub)

# #     def _pool(self, H, M, q):  # H:[B,L,D], M:[B,L], q:[h,D]
# #         # 다중 쿼리 평균 (안정성)
# #         Q = q.unsqueeze(0).unsqueeze(2)               # [1,h,1,D]
# #         Ht = self.proj(H)                              # [B,L,D]
# #         att = torch.einsum("bld, h1d -> bhl", Ht, q)  # [B,h,L]
# #         att = att.masked_fill(M[:,None,:]==0, -1e4)
# #         w = torch.softmax(att, dim=-1)                # [B,h,L]
# #         s = torch.einsum("bhl, bld -> bhd", w, H)     # [B,h,D]
# #         s = s.mean(1)                                 # [B,D]
# #         return s

# #     def forward(self, H, mask):
# #         base = self._pool(H, mask, self.q_base)
# #         add  = self._pool(H, mask, self.q_add)
# #         rem  = self._pool(H, mask, self.q_remove)

# #         # 직교화: add/rem 성분에서 base 방향 제거
# #         def deproj(x, b):
# #             b = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #             return x - (x*b).sum(-1, keepdim=True) * b

# #         add  = deproj(add, base);  add  = self.P_add(add)
# #         rem  = deproj(rem, base);  rem  = self.P_rem(rem)

# #         return {"base": base, "add": add, "remove": rem}

# # # Sequential로는 residual connection이 어려우니 별도 클래스로
# # class FiveGate(nn.Module):
# #     def __init__(self, hidden_size):
# #         super().__init__()
# #         self.layer1 = nn.Linear(hidden_size * 5, hidden_size)
# #         self.layer2 = nn.Linear(hidden_size, hidden_size)
# #         self.layer3 = nn.Linear(hidden_size, hidden_size)
# #         self.output = nn.Linear(hidden_size, 5)
# #         self.relu = nn.ReLU(inplace=True)
# #         self.sigmoid = nn.Sigmoid()
    
# #     def forward(self, x):
# #         x = self.relu(self.layer1(x))
# #         residual = x
# #         x = self.relu(self.layer2(x))
# #         x = x + residual  # residual connection
# #         x = self.relu(self.layer3(x))
# #         return self.sigmoid(self.output(x))

# # modules/modeling.py
# from __future__ import absolute_import, division, print_function, unicode_literals

# import logging
# import torch
# from torch import nn
# import numpy as np

# from modules.until_module import PreTrainedModel, AllGather, CrossEn
# from modules.module_cross import CrossModel, CrossConfig, Transformer_Text, Transformer_Gate
# from modules.module_clip import CLIP, convert_weights

# logger = logging.getLogger(__name__)
# allgather = AllGather.apply


# def show_log(task_config, info):
#     if task_config is None or task_config.local_rank == 0:
#         logger.warning(info)


# def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
#     if hasattr(source_config, source_attr_name):
#         if default_value is None or getattr(source_config, source_attr_name) != default_value:
#             setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
#             show_log(source_config, "Set {}.{}: {}.".format(
#                 target_name, target_attr_name, getattr(target_config, target_attr_name)
#             ))
#     return target_config


# def check_attr(target_name, task_config):
#     return hasattr(task_config, target_name) and task_config.__dict__[target_name]


# class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
#     def __init__(self, cross_config, *inputs, **kwargs):
#         super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
#         self.cross_config = cross_config
#         self.clip = None
#         self.cross = None

#     @classmethod
#     def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):
#         task_config = kwargs.get("task_config", None)
#         if task_config is not None:
#             if not hasattr(task_config, "local_rank"):
#                 task_config.__dict__["local_rank"] = 0
#             elif task_config.local_rank == -1:
#                 task_config.local_rank = 0

#         if state_dict is None:
#             state_dict = {}

#         pretrained_clip_name = getattr(task_config, 'pretrained_clip_name', "ViT-B/32")
#         clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
#         for key, val in clip_state_dict.items():
#             new_key = "clip." + key
#             if new_key not in state_dict:
#                 state_dict[new_key] = val.clone()

#         cross_config, _ = CrossConfig.get_config(
#             cross_model_name, cache_dir, type_vocab_size,
#             state_dict=None, task_config=task_config
#         )
#         model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

#         # seq(LSTM/Transf) init
#         if model.sim_header in ["seqLSTM", "seqTransf"]:
#             contain_frame_position = any(key.find("frame_position_embeddings") > -1 for key in state_dict.keys())
#             if contain_frame_position is False:
#                 for key, val in clip_state_dict.items():
#                     if key == "positional_embedding":
#                         state_dict["frame_position_embeddings.weight"] = val.clone()
#                         continue
#                     if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
#                         num_layer = int(key.split(".")[2])
#                         if num_layer < task_config.cross_num_hidden_layers:
#                             state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
#                         if num_layer < task_config.audio_query_layers:
#                             state_dict[key.replace("transformer.", "transformer_Fusion.")] = val.clone()
#                             continue

#         if state_dict is not None:
#             model = cls.init_preweight(model, state_dict, task_config=task_config)
#         return model


# class CLIP4Clip(CLIP4ClipPreTrainedModel):
#     def __init__(self, cross_config, clip_state_dict, task_config):
#         super(CLIP4Clip, self).__init__(cross_config)
#         self.task_config = task_config
#         self.ignore_video_index = -1

#         assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings
#         self._stage_one = True
#         self._stage_two = False
#         show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

#         self.loose_type = False
#         if self._stage_one and check_attr('loose_type', self.task_config):
#             self.loose_type = True
#             show_log(task_config, "Test retrieval by loose type.")

#         vit = "visual.proj" in clip_state_dict
#         assert vit
#         vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
#         vision_layers = len([k for k in clip_state_dict.keys()
#                              if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
#         vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
#         grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
#         image_resolution = vision_patch_size * grid_size

#         embed_dim = clip_state_dict["text_projection"].shape[1]
#         context_length = clip_state_dict["positional_embedding"].shape[0]
#         vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
#         transformer_width = clip_state_dict["ln_final.weight"].shape[0]
#         transformer_heads = transformer_width // 64
#         transformer_layers = len({k.split(".")[2] for k in clip_state_dict.keys()
#                                   if k.startswith("transformer.resblocks.")})

#         self.linear_patch = getattr(task_config, "linear_patch", "2d")
#         self.hidden_size = cross_config.hidden_size

#         # Build CLIP
#         cut_top_layer = 0
#         self.clip = CLIP(
#             embed_dim,
#             image_resolution, vision_layers - cut_top_layer, vision_width, vision_patch_size,
#             context_length, vocab_size, transformer_width, transformer_heads,
#             transformer_layers - cut_top_layer,
#             linear_patch=self.linear_patch
#         ).float()
#         for key in ["input_resolution", "context_length", "vocab_size"]:
#             if key in clip_state_dict:
#                 del clip_state_dict[key]
#         convert_weights(self.clip)

#         # args
#         self.sim_header = getattr(self.task_config, "sim_header", "seqTransf")
#         self.lambda_ = self.task_config.temperature

#         # 단일 스위치
#         self.retrieval_mode = getattr(self.task_config, "retrieval_mode", "five_weighted")

#         # modules
#         num_query_token = 12
#         self.query_tokens = nn.Parameter(torch.zeros(cross_config.hidden_size, num_query_token))
#         nn.init.orthogonal_(self.query_tokens, 1.0)

#         if self.loose_type is False:
#             cross_config.max_position_embeddings = context_length
#             cross_config = update_attr(
#                 "cross_config", cross_config, "num_hidden_layers",
#                 self.task_config, "cross_num_hidden_layers"
#             )
#             self.cross = CrossModel(cross_config)
#             self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

#         if self.sim_header in ["seqLSTM", "seqTransf"]:
#             self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
#             self.query_position_embeddings = nn.Embedding(num_query_token, cross_config.hidden_size)

#         if self.sim_header == "seqTransf":
#             # gated fusion용 (기존 five_weighted 경로에서 유지)
#             self.transformer_Fusion = Transformer_Text(
#                 width=transformer_width,
#                 layers=self.task_config.audio_query_layers,
#                 heads=transformer_heads
#             )
#             self.transformerClip = Transformer_Gate(
#                 width=transformer_width,
#                 layers=self.task_config.cross_num_hidden_layers,
#                 heads=transformer_heads
#             )

#             # five weighted는 기존과 동일
#             # 792A_2_avg_AVT도 AVT에서 five weighted 쓰므로 gate 필요
#             if self.retrieval_mode in {"five_weighted", "792A_2_avg_AVT"}:
#                 self.five_gate = nn.Sequential(
#                     nn.Linear(self.hidden_size * 5, self.hidden_size),
#                     nn.ReLU(inplace=True),
#                     nn.Linear(self.hidden_size, 5),
#                     nn.Sigmoid()
#                 )

#         if self.sim_header == "seqLSTM":
#             self.lstm_visual = nn.LSTM(
#                 input_size=cross_config.hidden_size,
#                 hidden_size=cross_config.hidden_size,
#                 batch_first=True, bidirectional=False, num_layers=1
#             )

#         self.loss_fct_vis = CrossEn()
#         self.apply(self.init_weights)

#     # =========================
#     # pooling helpers
#     # =========================
#     def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
#         attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
#         attention_mask_un[:, 0, :] = 0.
#         sequence_output = sequence_output * attention_mask_un
#         denom = torch.sum(attention_mask_un, dim=1, dtype=torch.float).clamp(min=1e-6)
#         text_out = torch.sum(sequence_output, dim=1) / denom
#         return text_out

#     def _mean_pooling_for_similarity_visual(self, visual_output, video_mask):
#         video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
#         visual_output = visual_output * video_mask_un
#         video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
#         video_mask_un_sum[video_mask_un_sum == 0.] = 1.
#         video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
#         return video_out

#     def _global_from_audio(self, a, how="cls"):
#         # a: [B, Ta, D]
#         if how == "cls":
#             g = a[:, 0, :]
#         else:
#             g = a.mean(dim=1)
#         return g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)

#     def get_text_global(self, input_ids, token_type_ids, attention_mask):
#         seq = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=False)
#         text_global = self._mean_pooling_for_similarity_sequence(seq, attention_mask)
#         valid = (attention_mask.sum(dim=1) > 0).float().unsqueeze(-1)
#         text_global = text_global * valid
#         norm = text_global.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#         text_global = (text_global / norm) * valid
#         return text_global

#     def _stack_text_globals_from_slots(self, ids, seg, mask):
#         # ids/seg/mask: [B, S, L] (S=4)
#         B, S, _ = ids.shape
#         outs = []
#         valids = []
#         for s in range(S):
#             m = mask[:, s, :]
#             valid = (m.sum(dim=1) > 0).float().unsqueeze(-1)
#             if valid.any():
#                 g = self.get_text_global(ids[:, s, :], seg[:, s, :], m)
#             else:
#                 g = torch.zeros(B, self.hidden_size, device=ids.device, dtype=torch.float)
#             outs.append(g)
#             valids.append(valid)
#         G = torch.stack(outs, dim=1)   # [B,S,D]
#         V = torch.stack(valids, dim=1) # [B,S,1]
#         num = V.sum(dim=1).clamp(min=1.0)  # [B,1]
#         mean_g = (G * V).sum(dim=1) / num
#         mean_g = mean_g / mean_g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#         return mean_g

#     # =========================
#     # low-level encoders
#     # =========================
#     def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
#         if shaped is False:
#             input_ids = input_ids.view(-1, input_ids.shape[-1])
#             token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
#             attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
#         bs_pair = input_ids.size(0)
#         if input_ids.shape[1] != 20:
#             sequence_hidden = self.clip.encode_text(input_ids).float()
#             sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))
#         else:
#             sequence_hidden = [self.clip.encode_text(input_ids[:, i, :]).float() for i in range(input_ids.shape[1])]
#             sequence_hidden = torch.stack(sequence_hidden, dim=1)
#         return sequence_hidden

#     def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
#         if shaped is False:
#             video_mask = video_mask.view(-1, video_mask.shape[-1])
#             video = torch.as_tensor(video).float()
#             b, pair, bs, ts, channel, h, w = video.shape
#             video = video.view(b * pair * bs * ts, channel, h, w)
#             video_frame = bs * ts
#         bs_pair = video_mask.size(0)
#         visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
#         visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
#         return visual_hidden

#     def get_audio_output(self, audio):
#         audio = audio.squeeze(1)
#         audio_hidden = self.clip.encode_audio(audio).float()
#         return audio_hidden

#     # =========================
#     # fusion blocks
#     # =========================
#     def _fuse_av_only(self, visual_output, video_mask, audio_output):
#         query_tokens = self.query_tokens.t().unsqueeze(0)
#         query_embed = query_tokens.expand(visual_output.shape[0], -1, -1)

#         position_ids = torch.arange(query_embed.size(1), dtype=torch.long, device=visual_output.device)
#         position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
#         query_position_embeddings = self.query_position_embeddings(position_ids)
#         query_embed = query_embed + query_position_embeddings

#         query_mask = torch.ones(visual_output.shape[0], query_embed.shape[1], device=visual_output.device)
#         extended_query_mask = (1.0 - query_mask.unsqueeze(-1)) * -1000000.0
#         extended_query_mask = extended_query_mask.expand(-1, -1, audio_output.size(1))

#         query_embed = query_embed.permute(1, 0, 2)
#         audio_output = audio_output.permute(1, 0, 2)
#         qa_output = self.transformer_Fusion(query_embed, audio_output, extended_query_mask)
#         qa_output = qa_output.permute(1, 0, 2).contiguous()
#         audio_output = audio_output.permute(1, 0, 2).contiguous()

#         seq_length = visual_output.size(1)
#         position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
#         position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
#         frame_position_embeddings = self.frame_position_embeddings(position_ids)

#         visual_output_original = visual_output
#         visual_output = visual_output + frame_position_embeddings

#         extended_video_mask = (1.0 - video_mask.unsqueeze(-1)) * -1000000.0
#         extended_video_mask = extended_video_mask.expand(-1, -1, qa_output.size(1))

#         qa_output = qa_output.permute(1, 0, 2)
#         visual_output = visual_output.permute(1, 0, 2)
#         fusion_output, _, _, attn_gate_list, ff_gate_list = self.transformerClip(
#             visual_output, qa_output, extended_video_mask
#         )
#         fusion_output = fusion_output.permute(1, 0, 2)

#         fusion_output = 0.05 * fusion_output + 0.95 * visual_output_original
#         fusion_output = fusion_output.contiguous()

#         av_fused_output_global = self._mean_pooling_for_similarity_visual(fusion_output, video_mask)
#         av_fused_output_global = av_fused_output_global / av_fused_output_global.norm(dim=-1, keepdim=True).clamp(min=1e-6)

#         attn_gate_tensor = torch.stack(attn_gate_list, 1) if isinstance(attn_gate_list, (list, tuple)) else None
#         ff_gate_tensor = torch.stack(ff_gate_list, 1) if isinstance(ff_gate_list, (list, tuple)) else None
#         return av_fused_output_global, attn_gate_tensor, ff_gate_tensor

#     def _fuse_av_avg(self, visual_output, video_mask, audio_output):
#         # V global
#         v_g = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
#         v_g = v_g / v_g.norm(dim=-1, keepdim=True).clamp(min=1e-6)

#         # A global
#         a_g = self._global_from_audio(audio_output, how=getattr(self.task_config, "audio_pooling", "cls"))

#         # avg
#         av = 0.5 * (v_g + a_g)
#         av = av / av.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#         return av

#     # =========================
#     # five weighted
#     # =========================
#     def _five_weighted(self, av_g, obj_g, act_g, att_g, audm_g):
#         B, D = av_g.size()
#         cat = torch.cat([av_g, obj_g, act_g, att_g, audm_g], dim=-1)      # [B, 5D]
#         w = self.five_gate(cat)                                           # [B, 5]
#         comps = torch.stack([av_g, obj_g, act_g, att_g, audm_g], dim=1)   # [B, 5, D]
#         q = torch.sum(w.unsqueeze(-1) * comps, dim=1)                     # [B, D]
#         q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#         return q, w

#     # =========================
#     # Public APIs
#     # =========================
#     def encode_query(self, q_input_ids, q_token_type_ids, q_attention_mask, q_video, q_video_mask, q_audio):
#         if q_video_mask.dim() == 3 and q_video_mask.size(1) == 1:
#             q_video_mask = q_video_mask.squeeze(1)
#         elif q_video_mask.dim() != 2:
#             q_video_mask = q_video_mask.view(q_video_mask.size(0), -1)

#         rmode = self.retrieval_mode

#         # 입력: [B,4,L] 가정 (obj/act/att/audm)
#         if q_input_ids.dim() != 3 or q_input_ids.size(1) < 4:
#             raise ValueError("Current retrieval modes require q_input_ids shaped as [B,4,L]")

#         obj_g = self.get_text_global(q_input_ids[:, 0, :], q_token_type_ids[:, 0, :], q_attention_mask[:, 0, :])
#         act_g = self.get_text_global(q_input_ids[:, 1, :], q_token_type_ids[:, 1, :], q_attention_mask[:, 1, :])
#         att_g = self.get_text_global(q_input_ids[:, 2, :], q_token_type_ids[:, 2, :], q_attention_mask[:, 2, :])
#         audm_g = self.get_text_global(q_input_ids[:, 3, :], q_token_type_ids[:, 3, :], q_attention_mask[:, 3, :])

#         v_tok = self.get_visual_output(q_video, q_video_mask, shaped=False)
#         a_tok = self.get_audio_output(q_audio)

#         if rmode == "five_weighted":
#             av_g, _, _ = self._fuse_av_only(v_tok, q_video_mask, a_tok)
#             q, w = self._five_weighted(av_g, obj_g, act_g, att_g, audm_g)
#             self._cached_five_weights = w
#             return q

#         if rmode == "792A_2_avg_avg":
#             av_g = self._fuse_av_avg(v_tok, q_video_mask, a_tok)
#             t_mean = self._stack_text_globals_from_slots(q_input_ids, q_token_type_ids, q_attention_mask)
#             q = 0.5 * (av_g + t_mean)
#             q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#             self._cached_five_weights = None
#             return q

#         if rmode == "792A_2_avg_AVT":
#             av_g = self._fuse_av_avg(v_tok, q_video_mask, a_tok)
#             q, w = self._five_weighted(av_g, obj_g, act_g, att_g, audm_g)
#             self._cached_five_weights = w
#             return q
#         if rmode == "five_sum":
#             # AV는 gated fusion으로 만들고, 4개 텍스트 슬롯과 동일 가중치로 단순 합
#             av_g, _, _ = self._fuse_av_only(v_tok, q_video_mask, a_tok)

#             q = av_g + obj_g + act_g + att_g + audm_g
#             q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)

#             self._cached_five_weights = None
#             return q
#         raise ValueError("Unknown retrieval_mode: {}".format(rmode))

#     def encode_target(self, t_video, t_video_mask, t_audio):
#         if t_video_mask.dim() == 3 and t_video_mask.size(1) == 1:
#             t_video_mask = t_video_mask.squeeze(1)
#         elif t_video_mask.dim() != 2:
#             t_video_mask = t_video_mask.view(t_video_mask.size(0), -1)

#         v_tok = self.get_visual_output(t_video, t_video_mask, shaped=False)
#         a_tok = self.get_audio_output(t_audio)

#         # 792A_2_ 모드에서는 query/target 둘 다 avg
#         if self.retrieval_mode in {"792A_2_avg_avg", "792A_2_avg_AVT"}:
#             return self._fuse_av_avg(v_tok, t_video_mask, a_tok)

#         # 기존 five_weighted는 target은 gated fusion 유지
#         t_av_global, _, _ = self._fuse_av_only(v_tok, t_video_mask, a_tok)
#         return t_av_global

#     def forward(self,
#                 q_input_ids, q_token_type_ids, q_attention_mask,
#                 q_video, q_video_mask, q_audio,
#                 t_video, t_video_mask, t_audio):
#         assert self.sim_header == "seqTransf", "Gated Fusion(Transformer) required."

#         q_global = self.encode_query(q_input_ids, q_token_type_ids, q_attention_mask, q_video, q_video_mask, q_audio)
#         t_av_global = self.encode_target(t_video, t_video_mask, t_audio)
#         logit_scale = self.clip.logit_scale.exp()

#         if self.training:
#             q_all = allgather(q_global, self.task_config)
#             t_all = allgather(t_av_global, self.task_config)
#             torch.distributed.barrier()
#             sim_matrix = torch.matmul(q_all, t_all.t()) * logit_scale
#             loss = (self.loss_fct_vis(sim_matrix) + self.loss_fct_vis(sim_matrix.t())) / 2.0
#             return loss
#         else:
#             sim_matrix = torch.matmul(q_global, t_av_global.t()) * logit_scale
#             return sim_matrix, (None, None, None, None)


# modules/modeling.py
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import torch
from torch import nn
import numpy as np

from modules.until_module import PreTrainedModel, AllGather, CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer_Text, Transformer_Gate
from modules.module_clip import CLIP, convert_weights

logger = logging.getLogger(__name__)
allgather = AllGather.apply


def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)


def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(
                target_name, target_attr_name, getattr(target_config, target_attr_name)
            ))
    return target_config


def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):
        task_config = kwargs.get("task_config", None)
        if task_config is not None:
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None:
            state_dict = {}

        pretrained_clip_name = getattr(task_config, 'pretrained_clip_name', "ViT-B/32")
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(
            cross_model_name, cache_dir, type_vocab_size,
            state_dict=None, task_config=task_config
        )
        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        if model.sim_header in ["seqLSTM", "seqTransf"]:
            contain_frame_position = any(key.find("frame_position_embeddings") > -1 for key in state_dict.keys())
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                        if num_layer < task_config.audio_query_layers:
                            state_dict[key.replace("transformer.", "transformer_Fusion.")] = val.clone()
                            continue

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)
        return model


class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings
        self._stage_one = True
        self._stage_two = False
        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        vit = "visual.proj" in clip_state_dict
        assert vit
        vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in clip_state_dict.keys()
                             if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len({k.split(".")[2] for k in clip_state_dict.keys()
                                  if k.startswith("transformer.resblocks.")})

        self.linear_patch = getattr(task_config, "linear_patch", "2d")
        self.hidden_size = cross_config.hidden_size

        cut_top_layer = 0
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers - cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads,
            transformer_layers - cut_top_layer,
            linear_patch=self.linear_patch
        ).float()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]
        convert_weights(self.clip)

        self.sim_header = getattr(self.task_config, "sim_header", "seqTransf")
        self.lambda_ = self.task_config.temperature

        num_query_token = 12
        self.query_tokens = nn.Parameter(torch.zeros(cross_config.hidden_size, num_query_token))
        nn.init.orthogonal_(self.query_tokens, 1.0)

        if self.loose_type is False:
            cross_config.max_position_embeddings = context_length
            cross_config = update_attr(
                "cross_config", cross_config, "num_hidden_layers",
                self.task_config, "cross_num_hidden_layers"
            )
            self.cross = CrossModel(cross_config)
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header in ["seqLSTM", "seqTransf"]:
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
            self.query_position_embeddings = nn.Embedding(num_query_token, cross_config.hidden_size)

        if self.sim_header == "seqTransf":
            self.transformer_Fusion = Transformer_Text(
                width=transformer_width,
                layers=self.task_config.audio_query_layers,
                heads=transformer_heads
            )
            self.transformerClip = Transformer_Gate(
                width=transformer_width,
                layers=self.task_config.cross_num_hidden_layers,
                heads=transformer_heads
            )

            self.five_gate = nn.Sequential(
                nn.Linear(self.hidden_size * 5, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, 5),
                nn.Sigmoid()
            )

        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(
                input_size=cross_config.hidden_size,
                hidden_size=cross_config.hidden_size,
                batch_first=True, bidirectional=False, num_layers=1
            )

        self.loss_fct_vis = CrossEn()
        self.apply(self.init_weights)

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        denom = torch.sum(attention_mask_un, dim=1, dtype=torch.float).clamp(min=1e-6)
        text_out = torch.sum(sequence_output, dim=1) / denom
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _global_from_audio(self, a, how="cls"):
        if how == "cls":
            g = a[:, 0, :]
        else:
            g = a.mean(dim=1)
        return g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def get_text_global(self, input_ids, token_type_ids, attention_mask):
        seq = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=False)
        text_global = self._mean_pooling_for_similarity_sequence(seq, attention_mask)
        valid = (attention_mask.sum(dim=1) > 0).float().unsqueeze(-1)
        text_global = text_global * valid
        norm = text_global.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        text_global = (text_global / norm) * valid
        return text_global

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        bs_pair = input_ids.size(0)
        if input_ids.shape[1] != 20:
            sequence_hidden = self.clip.encode_text(input_ids).float()
            sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))
        else:
            sequence_hidden = [self.clip.encode_text(input_ids[:, i, :]).float() for i in range(input_ids.shape[1])]
            sequence_hidden = torch.stack(sequence_hidden, dim=1)
        return sequence_hidden

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts
        bs_pair = video_mask.size(0)
        visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
        return visual_hidden

    def get_audio_output(self, audio):
        audio = audio.squeeze(1)
        audio_hidden = self.clip.encode_audio(audio).float()
        return audio_hidden

    def _fuse_av_only(self, visual_output, video_mask, audio_output):
        query_tokens = self.query_tokens.t().unsqueeze(0)
        query_embed = query_tokens.expand(visual_output.shape[0], -1, -1)

        position_ids = torch.arange(query_embed.size(1), dtype=torch.long, device=visual_output.device)
        position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
        query_position_embeddings = self.query_position_embeddings(position_ids)
        query_embed = query_embed + query_position_embeddings

        query_mask = torch.ones(visual_output.shape[0], query_embed.shape[1], device=visual_output.device)
        extended_query_mask = (1.0 - query_mask.unsqueeze(-1)) * -1000000.0
        extended_query_mask = extended_query_mask.expand(-1, -1, audio_output.size(1))

        query_embed = query_embed.permute(1, 0, 2)
        audio_output = audio_output.permute(1, 0, 2)
        qa_output = self.transformer_Fusion(query_embed, audio_output, extended_query_mask)
        qa_output = qa_output.permute(1, 0, 2).contiguous()
        audio_output = audio_output.permute(1, 0, 2).contiguous()

        seq_length = visual_output.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
        position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)

        visual_output_original = visual_output
        visual_output = visual_output + frame_position_embeddings

        extended_video_mask = (1.0 - video_mask.unsqueeze(-1)) * -1000000.0
        extended_video_mask = extended_video_mask.expand(-1, -1, qa_output.size(1))

        qa_output = qa_output.permute(1, 0, 2)
        visual_output = visual_output.permute(1, 0, 2)
        fusion_output, _, _, attn_gate_list, ff_gate_list = self.transformerClip(
            visual_output, qa_output, extended_video_mask
        )
        fusion_output = fusion_output.permute(1, 0, 2)

        fusion_output = 0.05 * fusion_output + 0.95 * visual_output_original
        fusion_output = fusion_output.contiguous()

        av_fused_output_global = self._mean_pooling_for_similarity_visual(fusion_output, video_mask)
        av_fused_output_global = av_fused_output_global / av_fused_output_global.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        attn_gate_tensor = torch.stack(attn_gate_list, 1) if isinstance(attn_gate_list, (list, tuple)) else None
        ff_gate_tensor = torch.stack(ff_gate_list, 1) if isinstance(ff_gate_list, (list, tuple)) else None
        return av_fused_output_global, attn_gate_tensor, ff_gate_tensor

    def _five_weighted(self, av_g, obj_g, act_g, att_g, audm_g):
        cat = torch.cat([av_g, obj_g, act_g, att_g, audm_g], dim=-1)
        w = self.five_gate(cat)
        comps = torch.stack([av_g, obj_g, act_g, att_g, audm_g], dim=1)
        q = torch.sum(w.unsqueeze(-1) * comps, dim=1)
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return q, w

    def encode_query(self, q_input_ids, q_token_type_ids, q_attention_mask, q_video, q_video_mask, q_audio):
        if q_video_mask.dim() == 3 and q_video_mask.size(1) == 1:
            q_video_mask = q_video_mask.squeeze(1)
        elif q_video_mask.dim() != 2:
            q_video_mask = q_video_mask.view(q_video_mask.size(0), -1)

        if q_input_ids.dim() != 3 or q_input_ids.size(1) < 4:
            raise ValueError("q_input_ids must be [B,4,L]")

        obj_g = self.get_text_global(q_input_ids[:, 0, :], q_token_type_ids[:, 0, :], q_attention_mask[:, 0, :])
        act_g = self.get_text_global(q_input_ids[:, 1, :], q_token_type_ids[:, 1, :], q_attention_mask[:, 1, :])
        att_g = self.get_text_global(q_input_ids[:, 2, :], q_token_type_ids[:, 2, :], q_attention_mask[:, 2, :])
        audm_g = self.get_text_global(q_input_ids[:, 3, :], q_token_type_ids[:, 3, :], q_attention_mask[:, 3, :])

        v_tok = self.get_visual_output(q_video, q_video_mask, shaped=False)
        a_tok = self.get_audio_output(q_audio)

        av_g, _, _ = self._fuse_av_only(v_tok, q_video_mask, a_tok)
        q, w = self._five_weighted(av_g, obj_g, act_g, att_g, audm_g)
        self._cached_five_weights = w
        return q

    def encode_target(self, t_video, t_video_mask, t_audio):
        if t_video_mask.dim() == 3 and t_video_mask.size(1) == 1:
            t_video_mask = t_video_mask.squeeze(1)
        elif t_video_mask.dim() != 2:
            t_video_mask = t_video_mask.view(t_video_mask.size(0), -1)

        v_tok = self.get_visual_output(t_video, t_video_mask, shaped=False)
        a_tok = self.get_audio_output(t_audio)

        t_av_global, _, _ = self._fuse_av_only(v_tok, t_video_mask, a_tok)
        return t_av_global

    def forward(self,
                q_input_ids, q_token_type_ids, q_attention_mask,
                q_video, q_video_mask, q_audio,
                t_video, t_video_mask, t_audio):
        assert self.sim_header == "seqTransf", "Gated fusion is required."

        q_global = self.encode_query(q_input_ids, q_token_type_ids, q_attention_mask, q_video, q_video_mask, q_audio)
        t_av_global = self.encode_target(t_video, t_video_mask, t_audio)
        logit_scale = self.clip.logit_scale.exp()

        if self.training:
            q_all = allgather(q_global, self.task_config)
            t_all = allgather(t_av_global, self.task_config)
            torch.distributed.barrier()
            sim_matrix = torch.matmul(q_all, t_all.t()) * logit_scale
            loss = (self.loss_fct_vis(sim_matrix) + self.loss_fct_vis(sim_matrix.t())) / 2.0
            return loss
        else:
            sim_matrix = torch.matmul(q_global, t_av_global.t()) * logit_scale
            return sim_matrix, (None, None, None, None)
