# Copyright (c) 2023, Tri Dao, Albert Gu.
import math
from typing import Optional
import hydra
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from omegaconf import DictConfig, OmegaConf
from einops import rearrange, repeat
from .utils import SinusoidalPosEmb
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

logger = logging.getLogger(__name__)


"""self attention encoder + mamba decoder"""
class DiffusionMamba(nn.Module):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

    def __init__(
            self,
            encoder: DictConfig,
            mamba: DictConfig,
            state_dim: int,
            action_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            goal_drop: float = 0.1,
            linear_output: bool = False,
    ):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder)
        self.mamba = hydra.utils.instantiate(mamba)

        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len - 1 + action_seq_len

        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # we need another embedding for the time
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.time_emb.to(self.device)
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.action_emb.to(self.device)
        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            actions,
            time,
            states,
            goals,
            uncond: Optional[bool] = False,
    ):

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the time embedding
        times = rearrange(time, 'b -> b 1')
        emb_t = self.time_emb(times)

        if self.goal_conditioned:

            if self.training:
                goals = self.mask_cond(goals)
            # we want to use unconditional sampling during clasisfier free guidance
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)

            goal_embed = self.tok_emb(goals)

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        action_embed = self.action_emb(actions)

        position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len + self.action_seq_len - 1), :]

        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])
        action_x = self.drop(action_embed + position_embeddings[:, (self.goal_seq_len + t - 1):, :])

        input_seq = torch.cat([emb_t, state_x, action_x], dim=1)

        encoder_out = self.encoder(input_seq)
        action_seq = encoder_out[:, (t+1):, :]

        mamba_output = self.mamba(action_seq)

        pred_actions = self.action_pred(mamba_output)

        return pred_actions

    def get_params(self):
        return self.parameters()


""" would take all the output from the encoder to the mamba model """
class DiffusionMambaV2(nn.Module):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

    def __init__(
            self,
            encoder: DictConfig,
            mamba: DictConfig,
            state_dim: int,
            action_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            goal_drop: float = 0.1,
            linear_output: bool = False,
    ):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder)
        self.mamba = hydra.utils.instantiate(mamba)

        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len - 1 + action_seq_len

        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # we need another embedding for the time
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.time_emb.to(self.device)
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.action_emb.to(self.device)
        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            actions,
            time,
            states,
            goals,
            uncond: Optional[bool] = False,
    ):

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the time embedding
        times = rearrange(time, 'b -> b 1')
        emb_t = self.time_emb(times)

        if self.goal_conditioned:

            if self.training:
                goals = self.mask_cond(goals)
            # we want to use unconditional sampling during clasisfier free guidance
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)

            goal_embed = self.tok_emb(goals)

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        action_embed = self.action_emb(actions)

        position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len + self.action_seq_len - 1), :]

        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])
        action_x = self.drop(action_embed + position_embeddings[:, (self.goal_seq_len + t - 1):, :])

        input_seq = torch.cat([emb_t, state_x, action_x], dim=1)

        encoder_out = self.encoder(input_seq)

        mamba_output = self.mamba(encoder_out)[:, (t+1):, :]

        pred_actions = self.action_pred(mamba_output)

        return pred_actions

    def get_params(self):
        return self.parameters()


""" decoder only model only has self attention"""
class DiffusionDecoderOnly(nn.Module):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

    def __init__(
            self,
            encoder: DictConfig,
            # mamba: DictConfig,
            state_dim: int,
            action_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            goal_drop: float = 0.1,
            linear_output: bool = False,
    ):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder)
        # self.mamba = hydra.utils.instantiate(mamba)

        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len - 1 + action_seq_len

        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # we need another embedding for the time
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.time_emb.to(self.device)
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.action_emb.to(self.device)
        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            actions,
            time,
            states,
            goals,
            uncond: Optional[bool] = False,
    ):

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the time embedding
        times = rearrange(time, 'b -> b 1')
        emb_t = self.time_emb(times)

        if self.goal_conditioned:

            if self.training:
                goals = self.mask_cond(goals)
            # we want to use unconditional sampling during clasisfier free guidance
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)

            goal_embed = self.tok_emb(goals)

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        action_embed = self.action_emb(actions)

        position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len + self.action_seq_len - 1), :]

        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])
        action_x = self.drop(action_embed + position_embeddings[:, (self.goal_seq_len + t - 1):, :])

        input_seq = torch.cat([emb_t, state_x, action_x], dim=1)

        encoder_out = self.encoder(input_seq)
        action_seq = encoder_out[:, (t+1):, :]

        # mamba_output = self.mamba(action_seq)

        pred_actions = self.action_pred(action_seq)

        return pred_actions

    def get_params(self):
        return self.parameters()


""" mamba only model"""
class DiffusionMambaOnly(nn.Module):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

    def __init__(
            self,
            mamba: DictConfig,
            state_dim: int,
            action_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            goal_drop: float = 0.1,
            linear_output: bool = False,
    ):
        super().__init__()

        self.mamba = hydra.utils.instantiate(mamba)

        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len - 1 + action_seq_len

        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # we need another embedding for the time
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.time_emb.to(self.device)
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.action_emb.to(self.device)
        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            actions,
            time,
            states,
            goals,
            uncond: Optional[bool] = False,
    ):

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the time embedding
        times = rearrange(time, 'b -> b 1')
        emb_t = self.time_emb(times)

        if self.goal_conditioned:

            if self.training:
                goals = self.mask_cond(goals)
            # we want to use unconditional sampling during clasisfier free guidance
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)

            goal_embed = self.tok_emb(goals)

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        action_embed = self.action_emb(actions)

        position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len + self.action_seq_len - 1), :]

        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])
        action_x = self.drop(action_embed + position_embeddings[:, (self.goal_seq_len + t - 1):, :])

        input_seq = torch.cat([emb_t, state_x, action_x], dim=1)

        mamba_output = self.mamba(input_seq)

        pred_actions = self.action_pred(mamba_output)[:, (t+1):, :]

        return pred_actions

    def get_params(self):
        return self.parameters()


class DiffusionMambaOnlyLanguage(nn.Module):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

    def __init__(
            self,
            mamba: DictConfig,
            state_dim: int,
            action_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            goal_drop: float = 0.1,
            linear_output: bool = False,
    ):
        super().__init__()

        self.mamba = hydra.utils.instantiate(mamba)

        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len + action_seq_len

        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # we need another embedding for the time
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.time_emb.to(self.device)
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.action_emb.to(self.device)
        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            actions,
            time,
            states,
            goals,
            uncond: Optional[bool] = False,
    ):

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the time embedding
        times = rearrange(time, 'b -> b 1')
        emb_t = self.time_emb(times)

        if self.goal_conditioned:

            if self.training:
                goals = self.mask_cond(goals)
            # we want to use unconditional sampling during clasisfier free guidance
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)

            goal_embed = self.tok_emb(goals)

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        action_embed = self.action_emb(actions)

        position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len + self.action_seq_len - 1), :]

        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])
        action_x = self.drop(action_embed + position_embeddings[:, (self.goal_seq_len + t - 1):, :])

        input_seq = torch.cat([emb_t, state_x, action_x], dim=1)

        mamba_output = self.mamba(input_seq)

        pred_actions = self.action_pred(mamba_output)[:, (t+1):, :]

        return pred_actions

    def get_params(self):
        return self.parameters()


""" two mamba: mamba encoder and mamba decoder"""
class DiffusionMambaEncDec(nn.Module):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

    def __init__(
            self,
            mamba_encoder: DictConfig,
            mamba_decoder: DictConfig,
            state_dim: int,
            action_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            goal_drop: float = 0.1,
            linear_output: bool = False,
    ):
        super().__init__()

        self.mamba_encoder = hydra.utils.instantiate(mamba_encoder)
        self.mamba_decoder = hydra.utils.instantiate(mamba_decoder)

        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len - 1 + action_seq_len

        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # we need another embedding for the time
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.time_emb.to(self.device)
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.action_emb.to(self.device)
        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            actions,
            time,
            states,
            goals,
            uncond: Optional[bool] = False,
    ):

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the time embedding
        times = rearrange(time, 'b -> b 1')
        emb_t = self.time_emb(times)

        if self.goal_conditioned:

            if self.training:
                goals = self.mask_cond(goals)
            # we want to use unconditional sampling during clasisfier free guidance
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)

            goal_embed = self.tok_emb(goals)

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        action_embed = self.action_emb(actions)

        position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len + self.action_seq_len - 1), :]

        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])
        action_x = self.drop(action_embed + position_embeddings[:, (self.goal_seq_len + t - 1):, :])

        # encoder_input = torch.cat([emb_t, state_x], dim=1)
        history_state = self.mamba_encoder(state_x)
        current_state = history_state[:, -1:, :]

        input_seq = torch.cat([emb_t, current_state, action_x], dim=1)

        mamba_output = self.mamba_decoder(input_seq)

        pred_actions = self.action_pred(mamba_output)[:, 2:, :]

        return pred_actions

    def get_params(self):
        return self.parameters()


""" mamba process the history + decoder contains self attention and causal cross attention"""
class DiffusionMambaHistory(nn.Module):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

    def __init__(
            self,
            decoder: DictConfig,
            mamba: DictConfig,
            state_dim: int,
            action_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            goal_drop: float = 0.1,
            linear_output: bool = False,
    ):
        super().__init__()

        self.decoder = hydra.utils.instantiate(decoder)
        self.mamba = hydra.utils.instantiate(mamba)

        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len - 1 + action_seq_len

        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # we need another embedding for the time
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.time_emb.to(self.device)
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.action_emb.to(self.device)
        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            actions,
            time,
            states,
            goals,
            uncond: Optional[bool] = False,
    ):

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the time embedding
        times = rearrange(time, 'b -> b 1')
        emb_t = self.time_emb(times)

        if self.goal_conditioned:

            if self.training:
                goals = self.mask_cond(goals)
            # we want to use unconditional sampling during clasisfier free guidance
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)

            goal_embed = self.tok_emb(goals)

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        action_embed = self.action_emb(actions)

        position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len + self.action_seq_len - 1), :]

        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])
        action_x = self.drop(action_embed + position_embeddings[:, (self.goal_seq_len + t - 1):, :])

        input_state = torch.cat([emb_t, state_x], dim=1)
        mamba_output = self.mamba(input_state)

        decoder_out = self.decoder(action_x, mamba_output)

        pred_actions = self.action_pred(decoder_out)

        return pred_actions

    def get_params(self):
        return self.parameters()


""" take the mamba processed time embedding and current state as input to the decoder"""
class DiffusionMambaHistoryV2(nn.Module):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

    def __init__(
            self,
            decoder: DictConfig,
            mamba: DictConfig,
            state_dim: int,
            action_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            goal_drop: float = 0.1,
            linear_output: bool = False,
    ):
        super().__init__()

        self.decoder = hydra.utils.instantiate(decoder)
        self.mamba = hydra.utils.instantiate(mamba)

        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len - 1 + action_seq_len

        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # we need another embedding for the time
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.time_emb.to(self.device)
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.action_emb.to(self.device)
        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            actions,
            time,
            states,
            goals,
            uncond: Optional[bool] = False,
    ):

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the time embedding
        times = rearrange(time, 'b -> b 1')
        emb_t = self.time_emb(times)

        if self.goal_conditioned:

            if self.training:
                goals = self.mask_cond(goals)
            # we want to use unconditional sampling during clasisfier free guidance
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)

            goal_embed = self.tok_emb(goals)

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        action_embed = self.action_emb(actions)

        position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len + self.action_seq_len - 1), :]

        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])
        action_x = self.drop(action_embed + position_embeddings[:, (self.goal_seq_len + t - 1):, :])

        input_state = torch.cat([emb_t, state_x], dim=1)
        mamba_output = self.mamba(input_state)

        current_state = mamba_output[:, -1, :].unsqueeze(1)
        time_e = mamba_output[:, 0, :].unsqueeze(1)
        cross_input = torch.cat([current_state, time_e], dim=1)

        encoder_out = self.decoder(action_x, cross_input)

        pred_actions = self.action_pred(encoder_out)

        return pred_actions

    def get_params(self):
        return self.parameters()


"""mamba -> action noise seq  ||  decoder(action noise + action noise seq) -> action"""
class DiffusionMambaEn(nn.Module):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

    def __init__(
            self,
            decoder: DictConfig,
            mamba: DictConfig,
            state_dim: int,
            action_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            goal_drop: float = 0.1,
            linear_output: bool = False,
    ):
        super().__init__()

        self.decoder = hydra.utils.instantiate(decoder)
        self.mamba = hydra.utils.instantiate(mamba)

        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len - 1 + action_seq_len

        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # we need another embedding for the time
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.time_emb.to(self.device)
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.action_emb.to(self.device)
        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            actions,
            time,
            states,
            goals,
            uncond: Optional[bool] = False,
    ):

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the time embedding
        times = rearrange(time, 'b -> b 1')
        emb_t = self.time_emb(times)

        if self.goal_conditioned:

            if self.training:
                goals = self.mask_cond(goals)
            # we want to use unconditional sampling during clasisfier free guidance
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)

            goal_embed = self.tok_emb(goals)

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        action_embed = self.action_emb(actions)

        position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len + self.action_seq_len - 1), :]

        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])
        action_x = self.drop(action_embed + position_embeddings[:, (self.goal_seq_len + t - 1):, :])

        input_seq = torch.cat([emb_t, state_x, action_x], dim=1)

        encoder_out = self.mamba(input_seq)
        action_seq = encoder_out[:, (t+1):, :]

        mamba_output = self.decoder(action_x, action_seq)

        pred_actions = self.action_pred(mamba_output)

        return pred_actions

    def get_params(self):
        return self.parameters()


"""take the mamba processed emb_t and current state and feed it to the decoder"""
class DiffusionMambaEnTime(nn.Module):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

    def __init__(
            self,
            decoder: DictConfig,
            mamba: DictConfig,
            state_dim: int,
            action_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            goal_drop: float = 0.1,
            linear_output: bool = False,
    ):
        super().__init__()

        self.decoder = hydra.utils.instantiate(decoder)
        self.mamba = hydra.utils.instantiate(mamba)

        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len - 1 + action_seq_len

        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # we need another embedding for the time
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.time_emb.to(self.device)
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.action_emb.to(self.device)
        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            actions,
            time,
            states,
            goals,
            uncond: Optional[bool] = False,
    ):

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the time embedding
        times = rearrange(time, 'b -> b 1')
        emb_t = self.time_emb(times)

        if self.goal_conditioned:

            if self.training:
                goals = self.mask_cond(goals)
            # we want to use unconditional sampling during clasisfier free guidance
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)

            goal_embed = self.tok_emb(goals)

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        action_embed = self.action_emb(actions)

        position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len + self.action_seq_len - 1), :]

        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])
        action_x = self.drop(action_embed + position_embeddings[:, (self.goal_seq_len + t - 1):, :])

        input_seq = torch.cat([emb_t, state_x, action_x], dim=1)

        encoder_out = self.mamba(input_seq)

        time_e = encoder_out[:, 0, :].unsqueeze(1)
        cur_state = encoder_out[:, (t+1), :].unsqueeze(1)
        action_seq = encoder_out[:, (t+1):, :]

        cross_input = torch.cat([time_e, cur_state, action_seq], dim=1)
        mamba_output = self.decoder(action_x, cross_input)

        pred_actions = self.action_pred(mamba_output)

        return pred_actions

    def get_params(self):
        return self.parameters()


"""take the raw emb_t and current state and feed it to the decoder"""
class DiffusionMambaEnV2(nn.Module):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

    def __init__(
            self,
            decoder: DictConfig,
            mamba: DictConfig,
            state_dim: int,
            action_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            goal_drop: float = 0.1,
            linear_output: bool = False,
    ):
        super().__init__()

        self.decoder = hydra.utils.instantiate(decoder)
        self.mamba = hydra.utils.instantiate(mamba)

        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len - 1 + action_seq_len

        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # we need another embedding for the time
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.time_emb.to(self.device)
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.action_emb.to(self.device)
        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            actions,
            time,
            states,
            goals,
            uncond: Optional[bool] = False,
    ):

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the time embedding
        times = rearrange(time, 'b -> b 1')
        emb_t = self.time_emb(times)

        if self.goal_conditioned:

            if self.training:
                goals = self.mask_cond(goals)
            # we want to use unconditional sampling during clasisfier free guidance
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)

            goal_embed = self.tok_emb(goals)

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        action_embed = self.action_emb(actions)

        position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len + self.action_seq_len - 1), :]

        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])
        action_x = self.drop(action_embed + position_embeddings[:, (self.goal_seq_len + t - 1):, :])
        cur_state = state_x[:, -1, :].unsqueeze(1)

        input_seq = torch.cat([emb_t, state_x, action_x], dim=1)

        encoder_out = self.mamba(input_seq)
        action_seq = encoder_out[:, (t+1):, :]

        coss_input = torch.cat([emb_t, cur_state, action_seq], dim=1)
        decoder_output = self.decoder(action_x, coss_input)

        pred_actions = self.action_pred(decoder_output)

        return pred_actions

    def get_params(self):
        return self.parameters()


""" mamba + transformer"""
class DiffusionEncDecMamba(nn.Module):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

    def __init__(
            self,
            encoder: DictConfig,
            decoder: DictConfig,
            mamba: DictConfig,
            state_dim: int,
            action_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            goal_drop: float = 0.1,
            linear_output: bool = False,
    ):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder)
        self.decoder = hydra.utils.instantiate(decoder)
        self.mamba = hydra.utils.instantiate(mamba)

        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len - 1 + action_seq_len

        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # we need another embedding for the time
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.time_emb.to(self.device)
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.action_emb.to(self.device)
        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            actions,
            time,
            states,
            goals,
            uncond: Optional[bool] = False,
    ):

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the time embedding
        times = rearrange(time, 'b -> b 1')
        emb_t = self.time_emb(times)

        if self.goal_conditioned:

            if self.training:
                goals = self.mask_cond(goals)
            # we want to use unconditional sampling during clasisfier free guidance
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)

            goal_embed = self.tok_emb(goals)

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        action_embed = self.action_emb(actions)

        position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len + self.action_seq_len - 1), :]

        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])
        action_x = self.drop(action_embed + position_embeddings[:, (self.goal_seq_len + t - 1):, :])

        input_seq = torch.cat([emb_t, state_x, action_x], dim=1)
        current_state = state_x[:, -1, :].unsqueeze(1)
        encoder_input = torch.cat([emb_t, current_state], dim=1)

        mamba_out = self.mamba(input_seq)
        action_seq = mamba_out[:, (t+1):, :]

        encoder_out = self.encoder(encoder_input)

        cross_input = torch.cat([encoder_out, action_seq], dim=1)
        decoder_output = self.decoder(action_x, cross_input)

        pred_actions = self.action_pred(decoder_output)

        return pred_actions

    def get_params(self):
        return self.parameters()
