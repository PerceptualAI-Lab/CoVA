# -*- coding: utf-8 -*-
"""
Video/Audio Augmentation for SimCLR
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class VideoAugmentation(nn.Module):
    """
    비디오 프레임에 대한 augmentation
    
    적용 가능한 augmentation:
    1. Random Horizontal Flip
    2. Color Jitter (brightness, contrast, saturation)
    3. Random Grayscale
    4. Gaussian Blur
    5. Random Crop & Resize
    6. Temporal Jittering (프레임 순서 약간 섞기)
    
    Args:
        p_flip: horizontal flip 확률 (default: 0.5)
        p_color: color jitter 확률 (default: 0.8)
        p_gray: grayscale 변환 확률 (default: 0.2)
        p_blur: gaussian blur 확률 (default: 0.5)
        p_temporal: temporal jittering 확률 (default: 0.3)
    """
    def __init__(self, 
                 p_flip=0.5, 
                 p_color=0.8, 
                 p_gray=0.2, 
                 p_blur=0.5,
                 p_temporal=0.3,
                 color_jitter_strength=0.5):
        super(VideoAugmentation, self).__init__()
        self.p_flip = p_flip
        self.p_color = p_color
        self.p_gray = p_gray
        self.p_blur = p_blur
        self.p_temporal = p_temporal
        self.color_strength = color_jitter_strength
    
    def random_horizontal_flip(self, video):
        """
        video: [B, T, C, 3, H, W] or [B, T, 1, 3, H, W]
        """
        if random.random() < self.p_flip:
            return torch.flip(video, dims=[-1])  # flip W dimension
        return video
    
    def color_jitter(self, video):
        """
        Brightness, contrast, saturation jittering
        video: [B, T, C, 3, H, W]
        """
        if random.random() > self.p_color:
            return video
        
        # Brightness
        brightness_factor = 1.0 + random.uniform(-self.color_strength, self.color_strength)
        video = video * brightness_factor
        
        # Contrast
        contrast_factor = 1.0 + random.uniform(-self.color_strength, self.color_strength)
        mean = video.mean(dim=(-3, -2, -1), keepdim=True)
        video = (video - mean) * contrast_factor + mean
        
        video = torch.clamp(video, 0, 1)
        return video
    
    def random_grayscale(self, video):
        """
        video: [B, T, C, 3, H, W]
        """
        if random.random() > self.p_gray:
            return video
        
        # RGB to grayscale weights
        weights = torch.tensor([0.299, 0.587, 0.114], device=video.device)
        weights = weights.view(1, 1, 1, 3, 1, 1)
        
        gray = (video * weights).sum(dim=-3, keepdim=True)
        gray = gray.expand_as(video)
        return gray
    
    def gaussian_blur(self, video):
        """
        video: [B, T, C, 3, H, W]
        """
        if random.random() > self.p_blur:
            return video
        
        # Simple box blur as approximation
        kernel_size = random.choice([3, 5])
        padding = kernel_size // 2
        
        B, T, C, ch, H, W = video.shape
        video = video.view(-1, ch, H, W)
        
        # Average pooling as blur
        video = F.avg_pool2d(video, kernel_size=kernel_size, stride=1, padding=padding)
        video = video.view(B, T, C, ch, H, W)
        
        return video
    
    def temporal_jitter(self, video, video_mask):
        """
        프레임 순서를 약간 섞거나, 일부 프레임을 반복/삭제
        video: [B, T, C, 3, H, W]
        video_mask: [B, T]
        """
        if random.random() > self.p_temporal:
            return video, video_mask
        
        B, T = video.shape[:2]
        
        # 방법 1: 인접 프레임 swap
        for b in range(B):
            valid_frames = video_mask[b].sum().int().item()
            if valid_frames > 2:
                # 랜덤하게 2개 인접 프레임 swap
                swap_idx = random.randint(0, valid_frames - 2)
                video[b, swap_idx], video[b, swap_idx + 1] = \
                    video[b, swap_idx + 1].clone(), video[b, swap_idx].clone()
        
        return video, video_mask
    
    def forward(self, video, video_mask=None):
        """
        Args:
            video: [B, T, C, 3, H, W] or [B, 1, T, 1, 3, H, W]
            video_mask: [B, T] or [B, 1, T]
        
        Returns:
            augmented_video, video_mask
        """
        original_shape = video.shape
        
        # Shape normalization
        if video.dim() == 7:  # [B, 1, T, 1, 3, H, W]
            video = video.squeeze(1)  # [B, T, 1, 3, H, W]
        
        if video_mask is not None and video_mask.dim() == 3:
            video_mask = video_mask.squeeze(1)
        
        # Apply augmentations
        video = self.random_horizontal_flip(video)
        video = self.color_jitter(video)
        video = self.random_grayscale(video)
        video = self.gaussian_blur(video)
        
        if video_mask is not None:
            video, video_mask = self.temporal_jitter(video, video_mask)
        
        # Restore original shape
        if len(original_shape) == 7:
            video = video.unsqueeze(1)
            if video_mask is not None:
                video_mask = video_mask.unsqueeze(1)
        
        return video, video_mask


class AudioAugmentation(nn.Module):
    """
    오디오 (fbank) 에 대한 augmentation
    
    적용 가능한 augmentation:
    1. Time Masking (SpecAugment)
    2. Frequency Masking (SpecAugment)
    3. Additive Noise
    4. Time Warping
    5. Mixup (배치 내 다른 샘플과 mixing)
    
    Args:
        p_time_mask: time masking 확률 (default: 0.5)
        p_freq_mask: frequency masking 확률 (default: 0.5)
        p_noise: noise 추가 확률 (default: 0.3)
        time_mask_param: time mask 최대 길이 (default: 50)
        freq_mask_param: frequency mask 최대 길이 (default: 20)
    """
    def __init__(self,
                 p_time_mask=0.5,
                 p_freq_mask=0.5,
                 p_noise=0.3,
                 time_mask_param=50,
                 freq_mask_param=20,
                 noise_level=0.1):
        super(AudioAugmentation, self).__init__()
        self.p_time_mask = p_time_mask
        self.p_freq_mask = p_freq_mask
        self.p_noise = p_noise
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.noise_level = noise_level
    
    def time_masking(self, fbank):
        """
        SpecAugment Time Masking
        fbank: [B, 1, T, F] or [B, T, F]
        """
        if random.random() > self.p_time_mask:
            return fbank
        
        squeeze = False
        if fbank.dim() == 3:
            fbank = fbank.unsqueeze(1)
            squeeze = True
        
        B, _, T, F = fbank.shape
        
        for b in range(B):
            t = random.randint(1, min(self.time_mask_param, T // 4))
            t0 = random.randint(0, T - t)
            fbank[b, :, t0:t0+t, :] = 0
        
        if squeeze:
            fbank = fbank.squeeze(1)
        
        return fbank
    
    def freq_masking(self, fbank):
        """
        SpecAugment Frequency Masking
        fbank: [B, 1, T, F] or [B, T, F]
        """
        if random.random() > self.p_freq_mask:
            return fbank
        
        squeeze = False
        if fbank.dim() == 3:
            fbank = fbank.unsqueeze(1)
            squeeze = True
        
        B, _, T, F = fbank.shape
        
        for b in range(B):
            f = random.randint(1, min(self.freq_mask_param, F // 4))
            f0 = random.randint(0, F - f)
            fbank[b, :, :, f0:f0+f] = 0
        
        if squeeze:
            fbank = fbank.squeeze(1)
        
        return fbank
    
    def add_noise(self, fbank):
        """
        Gaussian noise 추가
        fbank: [B, 1, T, F] or [B, T, F]
        """
        if random.random() > self.p_noise:
            return fbank
        
        noise = torch.randn_like(fbank) * self.noise_level
        return fbank + noise
    
    def forward(self, fbank):
        """
        Args:
            fbank: [B, 1, T, F] - mel filterbank features
        
        Returns:
            augmented_fbank
        """
        fbank = self.time_masking(fbank)
        fbank = self.freq_masking(fbank)
        fbank = self.add_noise(fbank)
        
        return fbank


class JointAugmentation(nn.Module):
    """
    Video와 Audio를 함께 augment하는 wrapper
    
    SimCLR에서는 같은 샘플에 대해 두 개의 다른 augmented view를 생성해야 함
    이 클래스는 video와 audio에 대해 일관된 augmentation을 적용
    """
    def __init__(self, 
                 video_aug_config=None, 
                 audio_aug_config=None,
                 enable_video=True,
                 enable_audio=True):
        super(JointAugmentation, self).__init__()
        
        self.enable_video = enable_video
        self.enable_audio = enable_audio
        
        video_cfg = video_aug_config or {}
        audio_cfg = audio_aug_config or {}
        
        self.video_aug = VideoAugmentation(**video_cfg) if enable_video else None
        self.audio_aug = AudioAugmentation(**audio_cfg) if enable_audio else None
    
    def forward(self, video, video_mask, audio):
        """
        Args:
            video: [B, 1, T, 1, 3, H, W]
            video_mask: [B, 1, T]
            audio: [B, 1, T_a, F]
        
        Returns:
            aug_video, aug_video_mask, aug_audio
        """
        if self.video_aug is not None and self.enable_video:
            video, video_mask = self.video_aug(video, video_mask)
        
        if self.audio_aug is not None and self.enable_audio:
            audio = self.audio_aug(audio)
        
        return video, video_mask, audio


def create_simclr_augmentation(aug_strength='medium'):
    """
    Augmentation strength에 따른 preset 생성
    
    Args:
        aug_strength: 'weak', 'medium', 'strong'
    
    Returns:
        JointAugmentation instance
    """
    presets = {
        'weak': {
            'video': {
                'p_flip': 0.3,
                'p_color': 0.5,
                'p_gray': 0.1,
                'p_blur': 0.2,
                'p_temporal': 0.1,
                'color_jitter_strength': 0.3
            },
            'audio': {
                'p_time_mask': 0.3,
                'p_freq_mask': 0.3,
                'p_noise': 0.1,
                'time_mask_param': 30,
                'freq_mask_param': 10,
                'noise_level': 0.05
            }
        },
        'medium': {
            'video': {
                'p_flip': 0.5,
                'p_color': 0.8,
                'p_gray': 0.2,
                'p_blur': 0.5,
                'p_temporal': 0.3,
                'color_jitter_strength': 0.5
            },
            'audio': {
                'p_time_mask': 0.5,
                'p_freq_mask': 0.5,
                'p_noise': 0.2,
                'time_mask_param': 50,
                'freq_mask_param': 20,
                'noise_level': 0.1
            }
        },
        'strong': {
            'video': {
                'p_flip': 0.5,
                'p_color': 1.0,
                'p_gray': 0.3,
                'p_blur': 0.7,
                'p_temporal': 0.5,
                'color_jitter_strength': 0.8
            },
            'audio': {
                'p_time_mask': 0.7,
                'p_freq_mask': 0.7,
                'p_noise': 0.4,
                'time_mask_param': 80,
                'freq_mask_param': 30,
                'noise_level': 0.15
            }
        }
    }
    
    cfg = presets.get(aug_strength, presets['medium'])
    return JointAugmentation(
        video_aug_config=cfg['video'],
        audio_aug_config=cfg['audio']
    )