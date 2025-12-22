# from __future__ import absolute_import, division, unicode_literals, print_function

# import torch
# from torch.utils.data import DataLoader, Dataset

# from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader, MSRVTT_TrainDataLoader
# from dataloaders.dataloader_vatex_retrieval import VATEX_DataLoader
# from dataloaders.dataloader_charades_retrieval import Charades_DataLoader
# from dataloaders.dataloader_custom_retrieval import get_custom_train_loader, get_custom_test_loader

# class _QueryTargetPairDataset(Dataset):
#     """
#     기존 6-튜플:
#       (input_ids, input_mask, segment_ids, video, video_mask, audio, [extras...])
#     을 9-튜플로 확장:
#       (q_input_ids, q_input_mask, q_segment_ids, q_video, q_video_mask, q_audio,
#        t_video,     t_video_mask,     t_audio, [extras...])
#     """
#     def __init__(self, base_dataset):
#         self.base = base_dataset

#     def __len__(self):
#         return len(self.base)

#     def __getitem__(self, idx):
#         items = self.base[idx]
#         input_ids, input_mask, segment_ids, video, video_mask, audio = items[:6]
#         extras = items[6:]  # (video_id, sentence) 등 메타
#         return (input_ids, input_mask, segment_ids, video, video_mask, audio,
#                 video,     video_mask,     audio, *extras)


# def dataloader_msrvtt_train(args, tokenizer):
#     msrvtt_base = MSRVTT_TrainDataLoader(
#         csv_path=args.train_csv,
#         json_path=args.data_path,
#         features_path=args.features_path,
#         max_words=args.max_words,
#         feature_framerate=args.feature_framerate,
#         tokenizer=tokenizer,
#         max_frames=args.max_frames,
#         unfold_sentences=args.expand_msrvtt_sentences,
#         frame_order=args.train_frame_order,
#         slice_framepos=args.slice_framepos,
#         audio_path=args.audio_path,
#     )
#     msrvtt_dataset = _QueryTargetPairDataset(msrvtt_base)

#     train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
#     dataloader = DataLoader(
#         msrvtt_dataset,
#         batch_size=args.batch_size // args.n_gpu,
#         num_workers=args.num_thread_reader,
#         pin_memory=True,
#         shuffle=(train_sampler is None),
#         sampler=train_sampler,
#         drop_last=True,
#     )
#     return dataloader, len(msrvtt_dataset), train_sampler


# def dataloader_msrvtt_test(args, tokenizer, subset="test"):
#     msrvtt_test_base = MSRVTT_DataLoader(
#         csv_path=args.val_csv,
#         features_path=args.features_path,
#         max_words=args.max_words,
#         feature_framerate=args.feature_framerate,
#         tokenizer=tokenizer,
#         max_frames=args.eval_max_frames,
#         frame_order=args.eval_frame_order,
#         slice_framepos=args.slice_framepos,
#         audio_path=args.audio_path,
#     )
#     msrvtt_testset = _QueryTargetPairDataset(msrvtt_test_base)
#     dataloader_msrvtt = DataLoader(
#         msrvtt_testset,
#         batch_size=args.batch_size_val,
#         num_workers=args.num_thread_reader,
#         shuffle=False,
#         drop_last=False,
#     )
#     return dataloader_msrvtt, len(msrvtt_testset)


# def dataloader_vatex_train(args, tokenizer):
#     vatex_base = VATEX_DataLoader(
#         subset="train",
#         data_path=args.data_path,
#         features_path=args.features_path,
#         max_words=args.max_words,
#         feature_framerate=args.feature_framerate,
#         tokenizer=tokenizer,
#         max_frames=args.max_frames,
#         frame_order=args.train_frame_order,
#         slice_framepos=args.slice_framepos,
#         audio_path=args.audio_path,
#     )
#     vatex_dataset = _QueryTargetPairDataset(vatex_base)

#     train_sampler = torch.utils.data.distributed.DistributedSampler(vatex_dataset)
#     dataloader = DataLoader(
#         vatex_dataset,
#         batch_size=args.batch_size // args.n_gpu,
#         num_workers=args.num_thread_reader,
#         pin_memory=False,
#         shuffle=(train_sampler is None),
#         sampler=train_sampler,
#         drop_last=True,
#     )
#     return dataloader, len(vatex_dataset), train_sampler


# def dataloader_vatex_test(args, tokenizer, subset="test"):
#     vatex_test_base = VATEX_DataLoader(
#         subset=subset,
#         data_path=args.data_path,
#         features_path=args.features_path,
#         max_words=args.max_words,
#         feature_framerate=args.feature_framerate,
#         tokenizer=tokenizer,
#         max_frames=args.max_frames,
#         frame_order=args.eval_frame_order,
#         slice_framepos=args.slice_framepos,
#         audio_path=args.audio_path,
#     )
#     vatex_testset = _QueryTargetPairDataset(vatex_test_base)
#     dataloader_vatex = DataLoader(
#         vatex_testset,
#         batch_size=args.batch_size_val,
#         num_workers=args.num_thread_reader,
#         shuffle=False,
#         drop_last=False,
#     )
#     return dataloader_vatex, len(vatex_testset)


# def dataloader_charades_train(args, tokenizer):
#     charades_base = Charades_DataLoader(
#         csv_path=args.train_csv,
#         features_path=args.features_path,
#         max_words=args.max_words,
#         feature_framerate=args.feature_framerate,
#         tokenizer=tokenizer,
#         max_frames=args.max_frames,
#         frame_order=args.train_frame_order,
#         slice_framepos=args.slice_framepos,
#         audio_path=args.audio_path,
#     )
#     charades_dataset = _QueryTargetPairDataset(charades_base)

#     train_sampler = torch.utils.data.distributed.DistributedSampler(charades_dataset)
#     dataloader = DataLoader(
#         charades_dataset,
#         batch_size=args.batch_size // args.n_gpu,
#         num_workers=args.num_thread_reader,
#         pin_memory=True,
#         shuffle=(train_sampler is None),
#         sampler=train_sampler,
#         drop_last=True,
#     )
#     return dataloader, len(charades_dataset), train_sampler


# def dataloader_charades_test(args, tokenizer, subset="test"):
#     charades_test_base = Charades_DataLoader(
#         csv_path=args.val_csv,
#         features_path=args.features_path,
#         max_words=args.max_words,
#         feature_framerate=args.feature_framerate,
#         tokenizer=tokenizer,
#         max_frames=args.eval_max_frames,
#         frame_order=args.eval_frame_order,
#         slice_framepos=args.slice_framepos,
#         audio_path=args.audio_path,
#     )
#     charades_testset = _QueryTargetPairDataset(charades_test_base)
#     dataloader_charades = DataLoader(
#         charades_testset,
#         batch_size=args.batch_size_val,
#         num_workers=args.num_thread_reader,
#         shuffle=False,
#         drop_last=False,
#     )
#     return dataloader_charades, len(charades_testset)


# DATALOADER_DICT = {
#     "msrvtt":   {"train": dataloader_msrvtt_train,   "val": dataloader_msrvtt_test,   "test": None},
#     "vatex":    {"train": dataloader_vatex_train,    "val": dataloader_vatex_test,    "test": dataloader_vatex_test},
#     "charades": {"train": dataloader_charades_train, "val": dataloader_charades_test, "test": dataloader_charades_test},
#     "custom":   {"train": get_custom_train_loader,   "val": get_custom_test_loader,   "test": get_custom_test_loader},
# }


# dataloaders/data_dataloaders.py
from __future__ import absolute_import, division, unicode_literals, print_function

import torch
from torch.utils.data import DataLoader, Dataset

from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader, MSRVTT_TrainDataLoader
from dataloaders.dataloader_vatex_retrieval import VATEX_DataLoader
from dataloaders.dataloader_charades_retrieval import Charades_DataLoader
from dataloaders.dataloader_cova_retrieval import get_cova_train_loader, get_cova_test_loader


class _QueryTargetPairDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        items = self.base[idx]
        input_ids, input_mask, segment_ids, video, video_mask, audio = items[:6]
        extras = items[6:]
        return (input_ids, input_mask, segment_ids, video, video_mask, audio,
                video, video_mask, audio, *extras)


def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_base = MSRVTT_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        audio_path=args.audio_path,
    )
    msrvtt_dataset = _QueryTargetPairDataset(msrvtt_base)

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(msrvtt_dataset), train_sampler


def dataloader_msrvtt_test(args, tokenizer, subset="test"):
    msrvtt_test_base = MSRVTT_DataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.eval_max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        audio_path=args.audio_path,
    )
    msrvtt_testset = _QueryTargetPairDataset(msrvtt_test_base)
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)


def dataloader_vatex_train(args, tokenizer):
    vatex_base = VATEX_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        audio_path=args.audio_path,
    )
    vatex_dataset = _QueryTargetPairDataset(vatex_base)

    train_sampler = torch.utils.data.distributed.DistributedSampler(vatex_dataset)
    dataloader = DataLoader(
        vatex_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(vatex_dataset), train_sampler


def dataloader_vatex_test(args, tokenizer, subset="test"):
    vatex_test_base = VATEX_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        audio_path=args.audio_path,
    )
    vatex_testset = _QueryTargetPairDataset(vatex_test_base)
    dataloader_vatex = DataLoader(
        vatex_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_vatex, len(vatex_testset)


def dataloader_charades_train(args, tokenizer):
    charades_base = Charades_DataLoader(
        csv_path=args.train_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        audio_path=args.audio_path,
    )
    charades_dataset = _QueryTargetPairDataset(charades_base)

    train_sampler = torch.utils.data.distributed.DistributedSampler(charades_dataset)
    dataloader = DataLoader(
        charades_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(charades_dataset), train_sampler


def dataloader_charades_test(args, tokenizer, subset="test"):
    charades_test_base = Charades_DataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.eval_max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        audio_path=args.audio_path,
    )
    charades_testset = _QueryTargetPairDataset(charades_test_base)
    dataloader_charades = DataLoader(
        charades_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_charades, len(charades_testset)


DATALOADER_DICT = {
    "msrvtt":   {"train": dataloader_msrvtt_train,   "val": dataloader_msrvtt_test,   "test": None},
    "vatex":    {"train": dataloader_vatex_train,    "val": dataloader_vatex_test,    "test": dataloader_vatex_test},
    "charades": {"train": dataloader_charades_train, "val": dataloader_charades_test, "test": dataloader_charades_test},
    "cova":     {"train": get_cova_train_loader,     "val": get_cova_test_loader,     "test": get_cova_test_loader},
}
