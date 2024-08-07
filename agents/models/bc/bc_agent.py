import logging
import os
from collections import deque

import einops
import cv2
import ast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
from typing import Optional
from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class BC_Policy(nn.Module):

    def __init__(self,
                 model: DictConfig,
                 obs_encoder: DictConfig,
                 visual_input: bool = False,
                 device: str = 'cpu'):

        super(BC_Policy, self).__init__()

        self.visual_input = visual_input

        self.obs_encoder = hydra.utils.instantiate(obs_encoder).to(device)

        self.model = hydra.utils.instantiate(model).to(device)

    def forward(self, inputs):

        if self.visual_input:
            agentview_image, in_hand_image = inputs

            B, T, C, H, W = agentview_image.size()

            agentview_image = agentview_image.view(B * T, C, H, W)
            in_hand_image = in_hand_image.view(B * T, C, H, W)
            # state = state.view(B * T, -1)

            # bp_imgs = einops.rearrange(bp_imgs, "B T C H W -> (B T) C H W")
            # inhand_imgs = einops.rearrange(inhand_imgs, "B T C H W -> (B T) C H W")

            obs_dict = {"agentview_rgb": agentview_image,
                        "eye_in_hand_rgb": in_hand_image, }
            # "robot_ee_pos": state}

            obs = self.obs_encoder(obs_dict)
            obs = obs.view(B, T, -1)

        else:
            obs = self.obs_encoder(inputs)

        # make prediction
        pred = self.model(obs)

        return pred

    def get_params(self):
        return self.parameters()


class BC_Agent(BaseAgent):
    def __init__(
            self,
            model: DictConfig,
            optimization: DictConfig,
            trainset: DictConfig,
            valset: DictConfig,
            train_batch_size,
            val_batch_size,
            num_workers,
            device: str,
            epoch: int,
            obs_seq_len: int,
            action_seq_size: int,
            scale_data,
            eval_every_n_epochs: int = 50
    ):
        super().__init__(model=model, trainset=trainset, valset=valset, train_batch_size=train_batch_size,
                         val_batch_size=val_batch_size, num_workers=num_workers, device=device,
                         epoch=epoch, scale_data=scale_data, eval_every_n_epochs=eval_every_n_epochs)

        self.optimizer = hydra.utils.instantiate(
            optimization, params=self.model.parameters()
        )

        self.eval_model_name = "eval_best_bc.pth"
        self.last_model_name = "last_bc.pth"

        self.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
        self.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)

        self.obs_seq_len = obs_seq_len
        self.action_seq_size = action_seq_size
        self.action_counter = self.action_seq_size

        self.bp_image_context = deque(maxlen=self.obs_seq_len)
        self.inhand_image_context = deque(maxlen=self.obs_seq_len)

        self.train_loss = []
        self.test_mse = []

    def train_agent(self):
        best_test_mse = 1e10

        for num_epoch in tqdm(range(self.epoch)):

            if not (num_epoch + 1) % self.eval_every_n_epochs:
                # test_mse = []
                # for data in self.test_dataloader:
                #     state, action, mask = data  # [torch.squeeze(data[i]) for i in range(3)]
                #
                #     state = self.scaler.scale_input(state)
                #     action = self.scaler.scale_output(action)
                #
                #     mean_mse = self.evaluate(state, action)
                #     test_mse.append(mean_mse)
                #
                #     wandb.log(
                #         {
                #             "test_loss": mean_mse,
                #         }
                #     )
                self.eval_vision_agent()

                avrg_test_mse = sum(self.test_mse) / len(self.test_mse)

                log.info("Epoch {}: Mean test mse is {}".format(num_epoch, avrg_test_mse))

                if avrg_test_mse < best_test_mse:
                    best_test_mse = avrg_test_mse
                    self.store_model_weights(self.working_dir, sv_name=self.eval_model_name)

                    wandb.log(
                        {
                            "best_model_epochs": num_epoch
                        }
                    )

                    log.info('New best test loss. Stored weights have been updated!')

                wandb.log(
                    {
                        "mean_test_loss": avrg_test_mse,
                    }
                )

            self.train_vision_agent()
            # train_loss = []
            # for data in self.train_dataloader:
            #     state, action, mask = data  # [torch.squeeze(data[i]) for i in range(3)]
            #
            #     state = self.scaler.scale_input(state)
            #     action = self.scaler.scale_output(action)
            #
            #     batch_loss = self.train_step(state, action)
            #
            #     train_loss.append(batch_loss)
            #
            #     wandb.log(
            #         {
            #             "loss": batch_loss,
            #         }
            #     )

            avrg_train_loss = sum(self.train_loss) / len(self.train_loss)
            log.info("Epoch {}: Average train loss is {}".format(num_epoch, avrg_train_loss))

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)

        log.info("Training done!")

    def train_vision_agent(self):

        for data in self.train_dataloader:
            bp_imgs, inhand_imgs, action, mask = data

            bp_imgs = bp_imgs.to(self.device)
            inhand_imgs = inhand_imgs.to(self.device)

            # obs = self.scaler.scale_input(obs)
            action = self.scaler.scale_output(action)

            action = action[:, self.obs_seq_len - 1:, :].contiguous()

            # obs = obs[:, :self.obs_seq_len].contiguous()
            bp_imgs = bp_imgs[:, :self.obs_seq_len].contiguous()
            inhand_imgs = inhand_imgs[:, :self.obs_seq_len].contiguous()

            state = (bp_imgs, inhand_imgs)

            batch_loss = self.train_step(state, action)

            wandb.log({"train_loss": batch_loss.item()})

    def train_step(self, state, actions: torch.Tensor, goal: Optional[torch.Tensor] = None):
        """
        Executes a single training step on a mini-batch of data
        """
        self.model.train()

        out = self.model(state)
        loss = F.mse_loss(out, actions)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return loss

    @torch.no_grad()
    def eval_vision_agent(self):

        test_mse = []
        for data in self.test_dataloader:
            agentview_rgb, action, mask = data
            # obs, action, mask = data

            agentview_rgb = agentview_rgb.to(self.device)

            action = self.scaler.scale_output(action)

            batch_loss = self.evaluate(agentview_rgb, action)

            test_mse.append(batch_loss)

            wandb.log(
                {
                    "loss": batch_loss,
                }
            )

        self.test_mse = test_mse

    @torch.no_grad()
    def evaluate(self, state, action: torch.Tensor, goal: Optional[torch.Tensor] = None):
        """
        Method for evaluating the model on one epoch of data
        """
        self.model.eval()

        total_mse = 0.0

        if goal is not None:
            goal = self.scaler.scale_input(goal)
            out = self.model(torch.cat([state, goal], dim=-1))
        else:
            out = self.model(state)

        mse = F.mse_loss(out, action)  # , reduction="none")
        total_mse += mse.mean(dim=-1).sum().item()
        return total_mse

    def reset(self):
        """ Resets the context of the model."""
        self.action_counter = self.action_seq_size

        self.bp_image_context.clear()
        self.inhand_image_context.clear()

    @torch.no_grad()
    def predict(self, state, goal: Optional[torch.Tensor] = None, if_vision=True) -> torch.Tensor:
        """
        Method for predicting one step with input data
        """
        if if_vision:
            bp_image, inhand_image = state

            bp_image = torch.from_numpy(bp_image).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.
            inhand_image = torch.from_numpy(inhand_image).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.
            # des_robot_pos = torch.from_numpy(des_robot_pos).to(self.device).float().unsqueeze(0)

            # des_robot_pos = self.scaler.scale_input(des_robot_pos)

            self.bp_image_context.append(bp_image)
            self.inhand_image_context.append(inhand_image)
            # self.des_robot_pos_context.append(des_robot_pos)

            bp_image_seq = torch.stack(tuple(self.bp_image_context), dim=1)
            inhand_image_seq = torch.stack(tuple(self.inhand_image_context), dim=1)
            # des_robot_pos_seq = torch.stack(tuple(self.des_robot_pos_context), dim=1)

            input_state = (bp_image_seq, inhand_image_seq)
        else:
            obs = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            obs = self.scaler.scale_input(obs)
            self.obs_context.append(obs)
            input_state = torch.stack(tuple(self.obs_context), dim=1)  # type: ignore

        if self.action_counter == self.action_seq_size:
            self.action_counter = 0

            self.model.eval()

            model_pred = self.model(input_state)

            model_pred = self.scaler.inverse_scale_output(model_pred)

            self.curr_action_seq = model_pred

        next_action = self.curr_action_seq[:, self.action_counter, :]
        self.action_counter += 1
        return next_action.detach().cpu().numpy()

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """

        if sv_name is None:
            self.model.load_state_dict(torch.load(os.path.join(weights_path, "model_state_dict.pth")))
        else:
            self.model.load_state_dict(torch.load(os.path.join(weights_path, sv_name)))
        log.info('Loaded pre-trained model parameters')

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """

        if sv_name is None:
            torch.save(self.model.state_dict(), os.path.join(store_path, "model_state_dict.pth"))
        else:
            torch.save(self.model.state_dict(), os.path.join(store_path, sv_name))