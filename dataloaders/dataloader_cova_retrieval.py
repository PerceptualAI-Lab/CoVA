# # dataloaders/dataloader_custom_retrieval.py
# # -*- coding: utf-8 -*-
# from __future__ import absolute_import, division, unicode_literals, print_function

# import os
# import json
# from typing import Any, Dict, List

# import numpy as np
# import torch
# import torchaudio
# from torch.utils.data import Dataset, DataLoader, distributed

# from dataloaders.rawvideo_util import RawVideoExtractor

# __all__ = [
#     "CoVRCustomRetrieval",
#     "get_custom_train_loader",
#     "get_custom_test_loader",
#     "CoVRGalleryAV",
#     "_read_gallery_ids",
#     "get_custom_gallery_loader",
# ]

# SLOTS = ("object", "action", "attribute", "audio_mod_text")

# class CoVRCustomRetrieval(Dataset):

#     def __init__(self, split_json_path: str, tokenizer, features_path: str, audio_path: str,
#                  max_words: int = 32, max_frames: int = 12, feature_framerate: float = 1.0,
#                  image_resolution: int = 224, frame_order: int = 0, slice_framepos: int = 2):
#         self.data = json.load(open(split_json_path, "r", encoding="utf-8"))
#         self.features_path = features_path
#         self.audio_path = audio_path
#         self.tokenizer = tokenizer
#         self.max_words = max_words
#         self.max_frames = max_frames
#         self.frame_order = frame_order
#         self.slice_framepos = slice_framepos

#         self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
#         self.SPECIAL_TOKEN = {
#             "CLS_TOKEN": "<|startoftext|>",
#             "SEP_TOKEN": "<|endoftext|>",
#             "MASK_TOKEN": "[MASK]",
#             "UNK_TOKEN": "[UNK]",
#             "PAD_TOKEN": "[PAD]",
#         }

#         self.audio_sample_rate = 16000
#         self.audio_target_len = 1024
#         self.audio_norm_mean = -5.118
#         self.audio_norm_std  =  3.2527153

#         self._dbg_enable = os.getenv("COVR_DEBUG_PRINT_TEXT", "1") == "1"
#         self._dbg_limit  = int(os.getenv("COVR_DEBUG_MAX_PRINT", "5"))
#         self._dbg_count  = 0

#     def __len__(self): return len(self.data)

#     def _encode_text_single(self, text: str):

#         if text is None: text = ""
#         words = self.tokenizer.tokenize(text)
#         words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
#         total_len_with_cls = self.max_words - 1
#         if len(words) > total_len_with_cls:
#             words = words[:total_len_with_cls]
#         words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

#         input_ids = self.tokenizer.convert_tokens_to_ids(words)
#         input_mask = [1] * len(input_ids)
#         segment_ids = [0] * len(input_ids)

#         while len(input_ids) < self.max_words:
#             input_ids.append(0); input_mask.append(0); segment_ids.append(0)

#         # numpy (L,)
#         return (np.asarray(input_ids, dtype=np.int64),
#                 np.asarray(input_mask, dtype=np.int64),
#                 np.asarray(segment_ids, dtype=np.int64))

#     def _encode_text_slots4(self, item: Dict[str, Any]):

#         vmods = item.get("video_modification_text", None)
#         amod  = item.get("audio_modification_text", None)
#         base  = item.get("text", "")

#         slot_texts = {"object": "", "action": "", "attribute": "", "audio_mod_text": ""}

#         if isinstance(vmods, dict):
#             slot_texts["object"]    = (vmods.get("object") or "").strip()
#             slot_texts["action"]    = (vmods.get("action") or "").strip()
#             slot_texts["attribute"] = (vmods.get("attribute") or "").strip()
#         elif isinstance(vmods, str) and vmods.strip():
#             slot_texts["object"] = vmods.strip()
#         elif isinstance(base, str) and base.strip():
#             slot_texts["object"] = base.strip()

#         if isinstance(amod, str) and amod.strip():
#             slot_texts["audio_mod_text"] = amod.strip()

#         ids_list, mask_list, seg_list = [], [], []
#         for k in SLOTS:
#             ids, msk, seg = self._encode_text_single(slot_texts[k])
#             ids_list.append(ids)
#             mask_list.append(msk)
#             seg_list .append(seg)

#         # [4, L]
#         q_input_ids   = np.stack(ids_list,  axis=0)
#         q_input_mask  = np.stack(mask_list, axis=0)
#         q_segment_ids = np.stack(seg_list,  axis=0)

#         return q_input_ids, q_input_mask, q_segment_ids

#     def _resolve_video_path(self, vid: str) -> str:
#         for ext in ("mkv","mp4","webm"):
#             p = os.path.join(self.features_path, f"{vid}.{ext}")
#             if os.path.exists(p): return p
#         return os.path.join(self.features_path, f"{vid}.mkv")

#     def _get_rawvideo_single(self, vid: str):
#         video_mask = np.zeros((1, self.max_frames), dtype=np.int64)
#         video = np.zeros(
#             (1, self.max_frames, 1, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size),
#             dtype=np.float32
#         )
#         video_path = self._resolve_video_path(vid)
#         if not os.path.exists(video_path):
#             raise FileNotFoundError(f"Video file not found: id={vid} path={video_path}")

#         raw = self.rawVideoExtractor.get_video_data(video_path)
#         raw_video = raw.get('video', None)
#         if raw_video is None or raw_video.ndim <= 3:
#             raise ValueError(f"Invalid video: id={vid} path={video_path} shape={None if raw_video is None else raw_video.shape}")

#         raw_slice = self.rawVideoExtractor.process_raw_data(raw_video)
#         if raw_slice is None or raw_slice.ndim < 4:
#             raise ValueError(f"Failed to process video: id={vid} path={video_path}")

#         if raw_slice.shape[0] > self.max_frames:
#             if   self.slice_framepos == 0: video_slice = raw_slice[:self.max_frames, ...]
#             elif self.slice_framepos == 1: video_slice = raw_slice[-self.max_frames:, ...]
#             else:
#                 idx = np.linspace(0, raw_slice.shape[0]-1, num=self.max_frames, dtype=int)
#                 video_slice = raw_slice[idx, ...]
#         else:
#             video_slice = raw_slice

#         video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

#         if video_slice.ndim == 4 and video_slice.shape[-1] == 3 and video_slice.shape[1] != 3:
#             video_slice = np.transpose(video_slice, (0, 3, 1, 2))
#         if video_slice.ndim == 4 and video_slice.shape[1] == 3:
#             video_slice = np.expand_dims(video_slice, 1)

#         video_slice = np.asarray(video_slice, dtype=np.float32)

#         L = min(video_slice.shape[0], self.max_frames)
#         if L <= 0:
#             raise ValueError(f"No valid frames: id={vid} path={video_path}")
#         video[0, :L, ...] = video_slice[:L, ...]
#         video_mask[0, :L] = 1

#         if np.abs(video).sum() == 0:
#             raise ValueError(f"Video all zeros after processing: id={vid} path={video_path}")

#         return video, video_mask

#     def _get_rawaudio_single(self, vid: str):
#         T = self.audio_target_len
#         fbanks = torch.zeros((1, T, 128), dtype=torch.float32)
#         wav = os.path.join(self.audio_path, f"{vid}.wav")
#         if not os.path.exists(wav): return fbanks
#         waveform, sr = torchaudio.load(wav)
#         if sr != self.audio_sample_rate:
#             waveform = torchaudio.transforms.Resample(sr, self.audio_sample_rate)(waveform)
#         waveform = waveform - waveform.mean()
#         frame_shift_ms = waveform.shape[1] * 1000 / (self.audio_sample_rate * T)
#         fbank = torchaudio.compliance.kaldi.fbank(
#             waveform, htk_compat=True, sample_frequency=self.audio_sample_rate,
#             use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0,
#             frame_shift=frame_shift_ms
#         )
#         p = T - fbank.shape[0]
#         if p > 0:
#             fbank = torch.nn.ZeroPad2d((0,0,0,p))(fbank)
#         elif p < 0:
#             fbank = fbank[:T,:]
#         fbank = (fbank - self.audio_norm_mean) / (self.audio_norm_std * 2)
#         fbanks[0] = fbank
#         return fbanks

#     def __getitem__(self, idx: int):
#         item = self.data[idx]
#         qid, tid = item["query_id"], item["target_id"]

#         if self._dbg_enable and self._dbg_count < self._dbg_limit:
#             self._dbg_count += 1

#         q_input_ids, q_input_mask, q_segment_ids = self._encode_text_slots4(item)

#         q_video, q_video_mask = self._get_rawvideo_single(qid)
#         q_audio = self._get_rawaudio_single(qid)

#         t_video, t_video_mask = self._get_rawvideo_single(tid)
#         t_audio = self._get_rawaudio_single(tid)

#         # 기존 9개 + qid + tid = 11개
#         return (q_input_ids, q_input_mask, q_segment_ids,
#                 q_video, q_video_mask, q_audio,
#                 t_video, t_video_mask, t_audio,
#                 qid, tid)

#     # def __getitem__(self, idx: int):
#     #     item = self.data[idx]
#     #     qid, tid = item["query_id"], item["target_id"]

#     #     if self._dbg_enable and self._dbg_count < self._dbg_limit:
#     #         # v = item.get("video_modification_text", None)
#     #         # a = item.get("audio_modification_text", None)
#     #         # print(f"[DL/TEXT] idx={idx} pair_key={item.get('pair_key','?')} qid={qid} tid={tid}")
#     #         # if isinstance(v, dict):
#     #             # print(f"  video.object   : {(v.get('object') or '').strip()}")
#     #             # print(f"  video.action   : {(v.get('action') or '').strip()}")
#     #             # print(f"  video.attribute: {(v.get('attribute') or '').strip()}")
#     #         # elif isinstance(v, str):
#     #             # print(f"  video_modification_text(str): {v.strip()}")
#     #         # else:
#     #             # print(f"  video_modification_text: None")
#     #         # print(f"  audio.mod_text: {(a or '').strip()}\n")
#     #         self._dbg_count += 1

#     #     q_input_ids, q_input_mask, q_segment_ids = self._encode_text_slots4(item)

#     #     q_video, q_video_mask = self._get_rawvideo_single(qid)
#     #     q_audio = self._get_rawaudio_single(qid)

#     #     t_video, t_video_mask = self._get_rawvideo_single(tid)
#     #     t_audio = self._get_rawaudio_single(tid)

#     #     # q_input_* : [4, L] -> collate 후 [B, 4, L]
#     #     # q_video   : [1, T, 1, 3, H, W]
#     #     return (q_input_ids, q_input_mask, q_segment_ids,
#     #             q_video, q_video_mask, q_audio,
#     #             t_video, t_video_mask, t_audio)

# def get_custom_train_loader(args, tokenizer):
#     assert args.train_json is not None, "--train_json 경로 필요."
#     ds = CoVRCustomRetrieval(
#         split_json_path=args.train_json,
#         tokenizer=tokenizer,
#         features_path=args.features_path,
#         audio_path=args.audio_path,
#         max_words=args.max_words,
#         max_frames=args.max_frames,
#         feature_framerate=args.feature_framerate,
#         image_resolution=224,
#         frame_order=args.train_frame_order,
#         slice_framepos=args.slice_framepos,
#     )
#     sampler = distributed.DistributedSampler(ds, shuffle=True) if torch.distributed.is_initialized() else None
#     loader = DataLoader(
#         ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
#         num_workers=args.num_thread_reader, pin_memory=True, drop_last=True,
#     )
#     return loader, len(ds), sampler

# def get_custom_test_loader(args, tokenizer):
#     assert args.test_json is not None, "--test_json 경로가 필요."
#     ds = CoVRCustomRetrieval(
#         split_json_path=args.test_json,
#         tokenizer=tokenizer,
#         features_path=args.features_path,
#         audio_path=args.audio_path,
#         max_words=args.max_words,
#         max_frames=getattr(args, "eval_max_frames", args.max_frames),
#         feature_framerate=args.feature_framerate,
#         image_resolution=224,
#         frame_order=args.eval_frame_order,
#         slice_framepos=args.slice_framepos,
#     )
#     loader = DataLoader(
#         ds, batch_size=args.batch_size_val, shuffle=False, sampler=None,
#         num_workers=args.num_thread_reader, pin_memory=True, drop_last=False,
#     )
#     return loader, len(ds)

# class CoVRGalleryAV(Dataset):
#     def __init__(self, gallery_ids: List[str], features_path: str, audio_path: str,
#                  max_frames: int = 12, feature_framerate: float = 1.0,
#                  image_resolution: int = 224, frame_order: int = 0, slice_framepos: int = 2):
#         self.ids = list(gallery_ids)
#         self.features_path = features_path
#         self.audio_path = audio_path
#         self.max_frames = max_frames
#         self.frame_order = frame_order
#         self.slice_framepos = slice_framepos

#         self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
#         self.audio_sample_rate = 16000
#         self.audio_target_len = 1024
#         self.audio_norm_mean = -5.118
#         self.audio_norm_std  =  3.2527153

#     def __len__(self): return len(self.ids)

#     def _resolve_video_path(self, vid: str) -> str:
#         for ext in ("mkv","mp4","webm"):
#             p = os.path.join(self.features_path, f"{vid}.{ext}")
#             if os.path.exists(p): return p
#         return os.path.join(self.features_path, f"{vid}.mkv")

#     def _get_rawvideo_single(self, vid: str):
#         import numpy as np
#         video_mask = np.zeros((1, self.max_frames), dtype=np.int64)
#         video = np.zeros(
#             (1, self.max_frames, 1, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size),
#             dtype=np.float32
#         )
#         video_path = self._resolve_video_path(vid)
#         if not os.path.exists(video_path):
#             raise FileNotFoundError(f"[GALLERY] Video not found for '{vid}': {video_path}")

#         raw = self.rawVideoExtractor.get_video_data(video_path)
#         raw_video = raw.get('video', None)
#         if raw_video is None or raw_video.ndim <= 3:
#             raise ValueError(f"[GALLERY] Invalid video dims: {None if raw_video is None else raw_video.shape} (ID: {vid})")

#         raw_slice = self.rawVideoExtractor.process_raw_data(raw_video)
#         if raw_slice is None or raw_slice.ndim < 4:
#             raise ValueError(f"[GALLERY] Processed video invalid for {video_path} (ID: {vid})")

#         if raw_slice.shape[0] > self.max_frames:
#             if   self.slice_framepos == 0: video_slice = raw_slice[:self.max_frames, ...]
#             elif self.slice_framepos == 1: video_slice = raw_slice[-self.max_frames:, ...]
#             else:
#                 idx = np.linspace(0, raw_slice.shape[0]-1, num=self.max_frames, dtype=int)
#                 video_slice = raw_slice[idx, ...]
#         else:
#             video_slice = raw_slice

#         video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)
#         if video_slice.ndim == 4 and video_slice.shape[-1] == 3 and video_slice.shape[1] != 3:
#             video_slice = np.transpose(video_slice, (0, 3, 1, 2))
#         if video_slice.ndim == 4 and video_slice.shape[1] == 3:
#             video_slice = np.expand_dims(video_slice, 1)

#         video_slice = np.asarray(video_slice, dtype=np.float32)

#         L = min(video_slice.shape[0], self.max_frames)
#         if L <= 0: raise ValueError(f"[GALLERY] No valid frames extracted: {video_path} (ID: {vid})")
#         video[0, :L, ...] = video_slice[:L, ...]
#         video_mask[0, :L] = 1

#         if np.abs(video).sum() == 0:
#             raise ValueError(f"[GALLERY] Video zeros after processing: {video_path} (ID: {vid})")

#         return video, video_mask

#     def _get_rawaudio_single(self, vid: str):
#         T = self.audio_target_len
#         fbanks = torch.zeros((1, T, 128), dtype=torch.float32)
#         wav = os.path.join(self.audio_path, f"{vid}.wav")
#         if not os.path.exists(wav): return fbanks
#         waveform, sr = torchaudio.load(wav)
#         if sr != self.audio_sample_rate:
#             waveform = torchaudio.transforms.Resample(sr, self.audio_sample_rate)(waveform)
#         waveform = waveform - waveform.mean()
#         frame_shift_ms = waveform.shape[1] * 1000 / (self.audio_sample_rate * T)
#         fbank = torchaudio.compliance.kaldi.fbank(
#             waveform, htk_compat=True, sample_frequency=self.audio_sample_rate,
#             use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0,
#             frame_shift=frame_shift_ms
#         )
#         p = T - fbank.shape[0]
#         if p > 0:
#             fbank = torch.nn.ZeroPad2d((0,0,0,p))(fbank)
#         elif p < 0:
#             fbank = fbank[:T,:]
#         fbank = (fbank - self.audio_norm_mean) / (self.audio_norm_std * 2)
#         fbanks[0] = fbank
#         return fbanks

#     def __getitem__(self, idx: int):
#         vid = self.ids[idx]
#         t_video, t_video_mask = self._get_rawvideo_single(vid)
#         t_audio = self._get_rawaudio_single(vid)
#         return (t_video, t_video_mask, t_audio)

# def _read_gallery_ids(path: str) -> List[str]:
#     with open(path, "r", encoding="utf-8") as f:
#         text = f.read().strip()
#     if text.startswith("["):
#         arr = json.loads(text)
#         ids = [s.strip() for s in arr if isinstance(s, str) and s.strip()]
#     else:
#         ids = [line.strip() for line in text.splitlines() if line.strip()]
#     if len(ids) == 0:
#         raise ValueError(f"Gallery id list is empty: {path}")
#     return ids

# def get_custom_gallery_loader(args, tokenizer=None):
#     assert getattr(args, "gallery_json", None) is not None, "--gallery_json 경로 필요."
#     ids = _read_gallery_ids(args.gallery_json)
#     ds = CoVRGalleryAV(
#         gallery_ids=ids,
#         features_path=args.features_path,
#         audio_path=args.audio_path,
#         max_frames=getattr(args, "eval_max_frames", args.max_frames),
#         feature_framerate=args.feature_framerate,
#         image_resolution=224,
#         frame_order=args.eval_frame_order,
#         slice_framepos=args.slice_framepos,
#     )
#     loader = DataLoader(
#         ds, batch_size=args.batch_size_val, shuffle=False, sampler=None,
#         num_workers=args.num_thread_reader, pin_memory=True, drop_last=False,
#     )
#     return loader, len(ds)


# dataloaders/dataloader_cova_retrieval.py
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, unicode_literals, print_function

import os
import json
from typing import Any, Dict, List

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, distributed

from dataloaders.rawvideo_util import RawVideoExtractor

__all__ = [
    "CoVACoVARetrieval",
    "get_cova_train_loader",
    "get_cova_test_loader",
    "CoVAGalleryAV",
    "_read_gallery_ids",
    "get_cova_gallery_loader",
]

SLOTS = ("object", "action", "attribute", "audio_mod_text")


class CoVACoVARetrieval(Dataset):
    def __init__(self, split_json_path: str, tokenizer, features_path: str, audio_path: str,
                 max_words: int = 32, max_frames: int = 12, feature_framerate: float = 1.0,
                 image_resolution: int = 224, frame_order: int = 0, slice_framepos: int = 2):
        self.data = json.load(open(split_json_path, "r", encoding="utf-8"))
        self.features_path = features_path
        self.audio_path = audio_path
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_frames = max_frames
        self.frame_order = frame_order
        self.slice_framepos = slice_framepos

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "[UNK]",
            "PAD_TOKEN": "[PAD]",
        }

        self.audio_sample_rate = 16000
        self.audio_target_len = 1024
        self.audio_norm_mean = -5.118
        self.audio_norm_std = 3.2527153

    def __len__(self):
        return len(self.data)

    def _encode_text_single(self, text: str):
        if text is None:
            text = ""
        words = self.tokenizer.tokenize(text)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_len_with_cls = self.max_words - 1
        if len(words) > total_len_with_cls:
            words = words[:total_len_with_cls]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        return (np.asarray(input_ids, dtype=np.int64),
                np.asarray(input_mask, dtype=np.int64),
                np.asarray(segment_ids, dtype=np.int64))

    def _encode_text_slots4(self, item: Dict[str, Any]):
        vmods = item.get("video_modification_text", None)
        amod = item.get("audio_modification_text", None)
        base = item.get("text", "")

        slot_texts = {"object": "", "action": "", "attribute": "", "audio_mod_text": ""}

        if isinstance(vmods, dict):
            slot_texts["object"] = (vmods.get("object") or "").strip()
            slot_texts["action"] = (vmods.get("action") or "").strip()
            slot_texts["attribute"] = (vmods.get("attribute") or "").strip()
        elif isinstance(vmods, str) and vmods.strip():
            slot_texts["object"] = vmods.strip()
        elif isinstance(base, str) and base.strip():
            slot_texts["object"] = base.strip()

        if isinstance(amod, str) and amod.strip():
            slot_texts["audio_mod_text"] = amod.strip()

        ids_list, mask_list, seg_list = [], [], []
        for k in SLOTS:
            ids, msk, seg = self._encode_text_single(slot_texts[k])
            ids_list.append(ids)
            mask_list.append(msk)
            seg_list.append(seg)

        q_input_ids = np.stack(ids_list, axis=0)
        q_input_mask = np.stack(mask_list, axis=0)
        q_segment_ids = np.stack(seg_list, axis=0)
        return q_input_ids, q_input_mask, q_segment_ids

    def _resolve_video_path(self, vid: str) -> str:
        for ext in ("mkv", "mp4", "webm"):
            p = os.path.join(self.features_path, f"{vid}.{ext}")
            if os.path.exists(p):
                return p
        return os.path.join(self.features_path, f"{vid}.mkv")

    def _get_rawvideo_single(self, vid: str):
        video_mask = np.zeros((1, self.max_frames), dtype=np.int64)
        video = np.zeros(
            (1, self.max_frames, 1, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size),
            dtype=np.float32
        )
        video_path = self._resolve_video_path(vid)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: id={vid} path={video_path}")

        raw = self.rawVideoExtractor.get_video_data(video_path)
        raw_video = raw.get('video', None)
        if raw_video is None or raw_video.ndim <= 3:
            raise ValueError(f"Invalid video: id={vid} path={video_path} shape={None if raw_video is None else raw_video.shape}")

        raw_slice = self.rawVideoExtractor.process_raw_data(raw_video)
        if raw_slice is None or raw_slice.ndim < 4:
            raise ValueError(f"Failed to process video: id={vid} path={video_path}")

        if raw_slice.shape[0] > self.max_frames:
            if self.slice_framepos == 0:
                video_slice = raw_slice[:self.max_frames, ...]
            elif self.slice_framepos == 1:
                video_slice = raw_slice[-self.max_frames:, ...]
            else:
                idx = np.linspace(0, raw_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                video_slice = raw_slice[idx, ...]
        else:
            video_slice = raw_slice

        video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

        if video_slice.ndim == 4 and video_slice.shape[-1] == 3 and video_slice.shape[1] != 3:
            video_slice = np.transpose(video_slice, (0, 3, 1, 2))
        if video_slice.ndim == 4 and video_slice.shape[1] == 3:
            video_slice = np.expand_dims(video_slice, 1)

        video_slice = np.asarray(video_slice, dtype=np.float32)

        L = min(video_slice.shape[0], self.max_frames)
        if L <= 0:
            raise ValueError(f"No valid frames: id={vid} path={video_path}")
        video[0, :L, ...] = video_slice[:L, ...]
        video_mask[0, :L] = 1

        if np.abs(video).sum() == 0:
            raise ValueError(f"Video all zeros after processing: id={vid} path={video_path}")

        return video, video_mask

    def _get_rawaudio_single(self, vid: str):
        T = self.audio_target_len
        fbanks = torch.zeros((1, T, 128), dtype=torch.float32)
        wav = os.path.join(self.audio_path, f"{vid}.wav")
        if not os.path.exists(wav):
            return fbanks
        waveform, sr = torchaudio.load(wav)
        if sr != self.audio_sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.audio_sample_rate)(waveform)
        waveform = waveform - waveform.mean()
        frame_shift_ms = waveform.shape[1] * 1000 / (self.audio_sample_rate * T)
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=self.audio_sample_rate,
            use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0,
            frame_shift=frame_shift_ms
        )
        p = T - fbank.shape[0]
        if p > 0:
            fbank = torch.nn.ZeroPad2d((0, 0, 0, p))(fbank)
        elif p < 0:
            fbank = fbank[:T, :]
        fbank = (fbank - self.audio_norm_mean) / (self.audio_norm_std * 2)
        fbanks[0] = fbank
        return fbanks

    def __getitem__(self, idx: int):
        item = self.data[idx]
        qid, tid = item["query_id"], item["target_id"]

        q_input_ids, q_input_mask, q_segment_ids = self._encode_text_slots4(item)

        q_video, q_video_mask = self._get_rawvideo_single(qid)
        q_audio = self._get_rawaudio_single(qid)

        t_video, t_video_mask = self._get_rawvideo_single(tid)
        t_audio = self._get_rawaudio_single(tid)

        return (q_input_ids, q_input_mask, q_segment_ids,
                q_video, q_video_mask, q_audio,
                t_video, t_video_mask, t_audio,
                qid, tid)


def get_cova_train_loader(args, tokenizer):
    assert args.train_json is not None, "--train_json is required."
    ds = CoVACoVARetrieval(
        split_json_path=args.train_json,
        tokenizer=tokenizer,
        features_path=args.features_path,
        audio_path=args.audio_path,
        max_words=args.max_words,
        max_frames=args.max_frames,
        feature_framerate=args.feature_framerate,
        image_resolution=224,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )
    sampler = distributed.DistributedSampler(ds, shuffle=True) if torch.distributed.is_initialized() else None
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_thread_reader, pin_memory=True, drop_last=True,
    )
    return loader, len(ds), sampler


def get_cova_test_loader(args, tokenizer):
    assert args.test_json is not None, "--test_json is required."
    ds = CoVACoVARetrieval(
        split_json_path=args.test_json,
        tokenizer=tokenizer,
        features_path=args.features_path,
        audio_path=args.audio_path,
        max_words=args.max_words,
        max_frames=getattr(args, "eval_max_frames", args.max_frames),
        feature_framerate=args.feature_framerate,
        image_resolution=224,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size_val, shuffle=False, sampler=None,
        num_workers=args.num_thread_reader, pin_memory=True, drop_last=False,
    )
    return loader, len(ds)


class CoVAGalleryAV(Dataset):
    def __init__(self, gallery_ids: List[str], features_path: str, audio_path: str,
                 max_frames: int = 12, feature_framerate: float = 1.0,
                 image_resolution: int = 224, frame_order: int = 0, slice_framepos: int = 2):
        self.ids = list(gallery_ids)
        self.features_path = features_path
        self.audio_path = audio_path
        self.max_frames = max_frames
        self.frame_order = frame_order
        self.slice_framepos = slice_framepos

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.audio_sample_rate = 16000
        self.audio_target_len = 1024
        self.audio_norm_mean = -5.118
        self.audio_norm_std = 3.2527153

    def __len__(self):
        return len(self.ids)

    def _resolve_video_path(self, vid: str) -> str:
        for ext in ("mkv", "mp4", "webm"):
            p = os.path.join(self.features_path, f"{vid}.{ext}")
            if os.path.exists(p):
                return p
        return os.path.join(self.features_path, f"{vid}.mkv")

    def _get_rawvideo_single(self, vid: str):
        video_mask = np.zeros((1, self.max_frames), dtype=np.int64)
        video = np.zeros(
            (1, self.max_frames, 1, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size),
            dtype=np.float32
        )
        video_path = self._resolve_video_path(vid)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"[GALLERY] Video not found for '{vid}': {video_path}")

        raw = self.rawVideoExtractor.get_video_data(video_path)
        raw_video = raw.get('video', None)
        if raw_video is None or raw_video.ndim <= 3:
            raise ValueError(f"[GALLERY] Invalid video dims: {None if raw_video is None else raw_video.shape} (ID: {vid})")

        raw_slice = self.rawVideoExtractor.process_raw_data(raw_video)
        if raw_slice is None or raw_slice.ndim < 4:
            raise ValueError(f"[GALLERY] Processed video invalid for {video_path} (ID: {vid})")

        if raw_slice.shape[0] > self.max_frames:
            if self.slice_framepos == 0:
                video_slice = raw_slice[:self.max_frames, ...]
            elif self.slice_framepos == 1:
                video_slice = raw_slice[-self.max_frames:, ...]
            else:
                idx = np.linspace(0, raw_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                video_slice = raw_slice[idx, ...]
        else:
            video_slice = raw_slice

        video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)
        if video_slice.ndim == 4 and video_slice.shape[-1] == 3 and video_slice.shape[1] != 3:
            video_slice = np.transpose(video_slice, (0, 3, 1, 2))
        if video_slice.ndim == 4 and video_slice.shape[1] == 3:
            video_slice = np.expand_dims(video_slice, 1)

        video_slice = np.asarray(video_slice, dtype=np.float32)

        L = min(video_slice.shape[0], self.max_frames)
        if L <= 0:
            raise ValueError(f"[GALLERY] No valid frames extracted: {video_path} (ID: {vid})")
        video[0, :L, ...] = video_slice[:L, ...]
        video_mask[0, :L] = 1

        if np.abs(video).sum() == 0:
            raise ValueError(f"[GALLERY] Video zeros after processing: {video_path} (ID: {vid})")

        return video, video_mask

    def _get_rawaudio_single(self, vid: str):
        T = self.audio_target_len
        fbanks = torch.zeros((1, T, 128), dtype=torch.float32)
        wav = os.path.join(self.audio_path, f"{vid}.wav")
        if not os.path.exists(wav):
            return fbanks
        waveform, sr = torchaudio.load(wav)
        if sr != self.audio_sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.audio_sample_rate)(waveform)
        waveform = waveform - waveform.mean()
        frame_shift_ms = waveform.shape[1] * 1000 / (self.audio_sample_rate * T)
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=self.audio_sample_rate,
            use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0,
            frame_shift=frame_shift_ms
        )
        p = T - fbank.shape[0]
        if p > 0:
            fbank = torch.nn.ZeroPad2d((0, 0, 0, p))(fbank)
        elif p < 0:
            fbank = fbank[:T, :]
        fbank = (fbank - self.audio_norm_mean) / (self.audio_norm_std * 2)
        fbanks[0] = fbank
        return fbanks

    def __getitem__(self, idx: int):
        vid = self.ids[idx]
        t_video, t_video_mask = self._get_rawvideo_single(vid)
        t_audio = self._get_rawaudio_single(vid)
        return (t_video, t_video_mask, t_audio)


def _read_gallery_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if text.startswith("["):
        arr = json.loads(text)
        ids = [s.strip() for s in arr if isinstance(s, str) and s.strip()]
    else:
        ids = [line.strip() for line in text.splitlines() if line.strip()]
    if len(ids) == 0:
        raise ValueError(f"Gallery id list is empty: {path}")
    return ids


def get_cova_gallery_loader(args, tokenizer=None):
    assert getattr(args, "gallery_json", None) is not None, "--gallery_json is required."
    ids = _read_gallery_ids(args.gallery_json)
    ds = CoVAGalleryAV(
        gallery_ids=ids,
        features_path=args.features_path,
        audio_path=args.audio_path,
        max_frames=getattr(args, "eval_max_frames", args.max_frames),
        feature_framerate=args.feature_framerate,
        image_resolution=224,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size_val, shuffle=False, sampler=None,
        num_workers=args.num_thread_reader, pin_memory=True, drop_last=False,
    )
    return loader, len(ds)
