from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset

class PushTLowdimDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_path,  # zarr文件路径，用于加载数据
            horizon=1,  # 序列长度
            pad_before=0,  # 序列前填充长度
            pad_after=0,  # 序列后填充长度
            obs_key='keypoint',  # 观察数据的键名
            state_key='state',  # 状态数据的键名
            action_key='action',  # 动作数据的键名
            seed=42,  # 随机种子，用于reproducibility
            val_ratio=0.0,  # 验证集比例
            max_train_episodes=None  # 最大训练episode数，用于限制训练集大小
            ):
        # 调用父类的初始化方法
        super().__init__()
        
        # 从zarr文件中加载数据到replay buffer
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, state_key, action_key])

        # 生成验证集掩码
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        
        # 生成训练集掩码
        train_mask = ~val_mask
        
        # 如果指定了最大训练集大小，对训练集进行下采样
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        # 初始化序列采样器
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
            )
        
        # 保存各种参数
        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        keypoint = sample[self.obs_key]
        state = sample[self.state_key]
        agent_pos = state[:,:2]
        obs = np.concatenate([
            keypoint.reshape(keypoint.shape[0], -1), 
            agent_pos], axis=-1)

        data = {
            'obs': obs, # T, D_o
            'action': sample[self.action_key], # T, D_a
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
