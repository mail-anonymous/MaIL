import os
import logging
import random

import hydra
import numpy as np
import multiprocessing as mp
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)

print(torch.cuda.is_available())

OmegaConf.register_new_resolver(
    "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path="config", config_name="benchmark_libero_new.yaml")
def main(cfg: DictConfig) -> None:

    set_seed_everywhere(cfg.seed)

    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.group,
        mode="online",
        config=wandb.config
    )

    agent = hydra.utils.instantiate(cfg.agents)
    env_sim = hydra.utils.instantiate(cfg.simulation)

    job_num = hydra.core.hydra_config.HydraConfig.get().job.num

    num_cpu = mp.cpu_count()
    cpu_set = list(range(num_cpu))
    current_num = int(job_num % 4)
    assign_cpus = cpu_set[current_num * cfg.n_cores:current_num * cfg.n_cores + cfg.n_cores]

    for num_epoch in tqdm(range(agent.epoch)):

        agent.train_vision_agent()

        if num_epoch in [49]:

            env_sim.test_agent(agent, assign_cpus, epoch=num_epoch)
    agent.store_model_weights(agent.working_dir, sv_name=agent.last_model_name)
    log.info("done")

    wandb.finish()


if __name__ == "__main__":
    main()
