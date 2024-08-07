import logging
import pickle

import cv2
import h5py
import os
import torch
import numpy as np
from dataset.base_dataset import TrajectoryDataset
from agents.utils import sim_framework_path
from dataset.task_language_map import TaskIDMapping, IDLanguageMapping, IDClassifierMapping
import imgaug.parameters as iap
from imgaug import augmenters as iaa


def get_max_data_len(data_directory: os.PathLike):
    if os.path.exists(data_directory):
        data_dir = data_directory
    else:
        print("data_path is missing")

    max_data_len = 0

    f = h5py.File(data_dir, 'r')
    demos = f['data']
    num_demos = len(list(f["data"].keys()))

    for i in range(num_demos):
        demo_name = f'demo_{i}'
        state = demos[demo_name]['states']
        length = state.shape[0]

        if length > max_data_len:
            max_data_len = length

    f.close()

    return max_data_len


""" for the policy need: agentview, eye_in_hand """
class MultiTaskDataset(TrajectoryDataset):
    def __init__(
            self,
            data_directory: os.PathLike,
            task_suite,
            # data='train',
            obs_keys,  # low_dim or rgb
            obs_modalities,
            dataset_keys=None,  # [actions, dones, obs, rewards, states]
            filter_by_attribute=None,
            padding=True,
            device="cpu",
            obs_dim: int = 32,
            action_dim: int = 7,
            state_dim: int = 45,
            max_len_data: int = 136,
            window_size: int = 1,
            num_data: int = 10,
            data_aug=False,
            aug_factor=0.02
    ):
        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_len_data=max_len_data,
            window_size=window_size
        )

        logging.info("Loading Libero Dataset")

        self.data_aug = data_aug
        self.aug_factor = aug_factor

        self.obs_keys = obs_keys  # low_dim || rgb
        logging.info("The dataset is {}".format(self.obs_keys))  #show low_dim or rgb

        self.data_dir = [os.path.join(data_directory, file)
                         for file in os.listdir(data_directory) if file.endswith('.hdf5')]
        # self.data_dir.sort()
        # if len(self.data_dir) > 20:
        #     self.data_dir = self.data_dir[:30]

        self.obs_modalities = obs_modalities["obs"][self.obs_keys]
        logging.info("The obs_modalities list is {}".format(self.obs_modalities))

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.dataset_keys = dataset_keys  # [actions, dones, obs, rewards, states]
        self.filter_by_attribute = filter_by_attribute

        task_emb_dir = "task_embeddings/"

        with open(task_emb_dir + task_suite + ".pkl", 'rb') as f:
            tasks = pickle.load(f)
        # goal_dict = {}

        data_embs = []

        actions = []
        masks = []
        agentview_rgb = []
        eye_in_hand_rgb = []

        # goal_rgbs = []

        for data_dir in self.data_dir:
            filename = os.path.basename(data_dir).split('.')[0][:-5]
            # task_id = TaskIDMapping[filename]

            task_emb = tasks[filename]

            # goal_dict[filename] = []

            f = h5py.File(data_dir, 'r')

            # get the image's basic shape from demo_0
            if self.obs_keys == "rgb":
                H, W, C = f["data"]["demo_0"]["obs"][self.obs_modalities[0]].shape[1:]

            # determinate which demo should be loaded using demo_keys_list
            if filter_by_attribute is not None:
                self.demo_keys_list = [elem.decode("utf-8") for elem in
                                       np.array(f["mask/{}".format(filter_by_attribute)][:])]
            else:
                self.demo_keys_list = list(f["data"].keys())

            indices = np.argsort([int(elem[5:]) for elem in self.demo_keys_list])
            num_demos = len(self.demo_keys_list)

            # load the states and actions in demos according to demo_keys_list
            for i in indices[:num_data]:
                demo_name = f'demo_{i}'
                demo = f["data"][demo_name]
                demo_length = demo.attrs["num_samples"]

                zero_actions = np.zeros((1, self.max_len_data, self.action_dim), dtype=np.float32)
                zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)

                # action_data = demo['actions'][:]
                zero_actions[0, :demo_length, :] = demo['actions'][:]
                zero_mask[0, :demo_length] = 1

                # zero_agentview = np.zeros((1, self.max_len_data, H, W, C), dtype=np.float32)
                # zero_inhand = np.zeros((1, self.max_len_data, H, W, C), dtype=np.float32)

                # goal_view = demo['obs']['agentview_rgb'][-1:]

                agent_view = demo['obs']['agentview_rgb'][:]
                eye_in_hand = demo['obs']['eye_in_hand_rgb'][:]

                actions.append(zero_actions)
                masks.append(zero_mask)

                agentview_rgb.append(agent_view)
                eye_in_hand_rgb.append(eye_in_hand)

                data_embs.append(task_emb)

                # goal_dict[filename].append(goal_view)
                # goal_rgbs.append(goal_view)

            f.close()

        self.actions = torch.from_numpy(np.concatenate(actions)).to(device).float()  # shape: N, T, D
        self.masks = torch.from_numpy(np.concatenate(masks)).to(device).float()
        # self.agentview_rgb = torch.from_numpy(np.concatenate(agentview_rgb)).to(device).float()  # shape: N, T, H, W, C | N is the number of demos
        # self.eye_in_hand_rgb = torch.from_numpy(np.concatenate(eye_in_hand_rgb)).to(device)

        self.agentview_rgb = agentview_rgb
        self.eye_in_hand_rgb = eye_in_hand_rgb

        self.tasks = tasks
        self.data_embs = data_embs
        # self.goal_dict = goal_dict

        # self.goal_rgbs = goal_rgbs

        self.num_data = len(self.actions)
        self.slices = self.get_slices()

    def get_slices(self):  #Extract sample slices that meet certain conditions
        slices = []

        min_seq_length = np.inf
        for i in range(self.num_data):
            T = self.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)

            if T - self.window_size < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={self.window_size}")
            else:
                slices += [
                    (i, start, start + self.window_size) for start in range(T - self.window_size + 1)
                ]  # slice indices follow convention [start, end)

        return slices

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_all_observations(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        result = []
        # mask out invalid observations
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.agentview_rgb[i, :T, :])
        return torch.cat(result, dim=0)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):

        i, start, end = self.slices[idx]

        # goal_rgb = self.goal_rgbs[i]

        task_emb = self.data_embs[i]

        agentview_rgb = self.agentview_rgb[i][start:end]
        eye_in_hand_rgb = self.eye_in_hand_rgb[i][start:end]

        if self.data_aug is True:
            # cv2.imshow('ori', agentview_rgb[0])

            aug = iaa.arithmetic.ReplaceElementwise(
                iap.FromLowerResolution(iap.Binomial(self.aug_factor), size_px=8),
                [255])

            agentview_rgb = aug(images=agentview_rgb)

            # cv2.imshow('ori', agentview_rgb[0])
            # cv2.imshow('ori1', agentview_rgb[1])
            # cv2.imshow('ori2', agentview_rgb[2])
            # cv2.imshow('ori3', agentview_rgb[3])
            # cv2.waitKey(0)

        # goal_rgb = torch.from_numpy(goal_rgb).to(self.device).float().permute(0, 3, 1, 2) / 255.

        task_emb = task_emb.to(self.device).float()

        agentview_rgb = torch.from_numpy(agentview_rgb).to(self.device).float().permute(0, 3, 1, 2) / 255.
        eye_in_hand_rgb = torch.from_numpy(eye_in_hand_rgb).to(self.device).float().permute(0, 3, 1, 2) / 255.

        act = self.actions[i, start:end]
        mask = self.masks[i, start:end]

        return agentview_rgb, eye_in_hand_rgb, act, task_emb