import logging
import os
import random
import cv2
import numpy as np
import torch
import wandb
import robosuite
import multiprocessing as mp
from .base_sim import BaseSim
from libero.libero.envs import *
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import imgaug.parameters as iap
from imgaug import augmenters as iaa


log = logging.getLogger(__name__)


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


def process_image_input(img_tensor):
    # return (img_tensor / 255. - 0.5) * 2.
    return img_tensor / 255.

class MultiTaskSim(BaseSim):
    def __init__(self,
                 num_episode,
                 max_step_per_episode,
                 task_suite: str,
                 use_eye_in_hand: bool,
                 seed,
                 device,
                 render,
                 n_cores,
                 camera_shape: tuple,
                 data_aug: bool = False,
                 aug_factor: float = 0.02,
                 task_id: int = 0,
                 nms: float = 0.1):
        super().__init__(seed, device, render, n_cores)

        # data augmentation
        self.data_aug = data_aug
        self.aug_factor = aug_factor
        self.aug = iaa.arithmetic.ReplaceElementwise(iap.FromLowerResolution(iap.Binomial(self.aug_factor), size_px=8),
                                                     [255])

        # according to the task_id, load the corresponding bddl file
        self.task_suite = task_suite

        self.use_eye_in_hand = use_eye_in_hand
        self.render = render
        self.task_id = task_id

        self.num_episode = num_episode
        self.max_step_per_episode = max_step_per_episode

        self.success_rate = 0

    def eval_agent(self, agent, contexts, context_ind, success, pid, cpu_set):
        print(os.getpid(), cpu_set)
        assign_process_to_cpu(os.getpid(), cpu_set)

        # env_ids = []

        print(contexts)

        for i, context in enumerate(contexts):

            # if context not in env_ids:
            # env_ids.append(context)

            task_suite = benchmark.get_benchmark_dict()[self.task_suite]()

            task_bddl_file = task_suite.get_task_bddl_file_path(context)
            init_states = task_suite.get_task_init_states(context)

            env_args = {
                "bddl_file_name": task_bddl_file,
                "camera_heights": 128,
                "camera_widths": 128
            }

            env = OffScreenRenderEnv(**env_args)

            agent.reset()
            env.seed(self.seed)
            env.reset()
            obs = env.set_init_state(init_state=init_states[context_ind[i]])

            # dummy actions all zeros for initial physics simulation
            dummy = np.zeros(7)
            dummy[-1] = -1.0  # set the last action to -1 to open the gripper
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)

            # multiprocessing simulation
            for j in range(self.max_step_per_episode):
                agentview_rgb = obs["agentview_image"]

                if self.data_aug:
                    agentview_rgb = self.aug(image=agentview_rgb)

                if self.use_eye_in_hand:
                    eye_in_hand_rgb = obs["robot0_eye_in_hand_image"]
                    state = (agentview_rgb, eye_in_hand_rgb)
                else:
                    state = agentview_rgb

                action = agent.predict(state)[0]
                obs, r, done, _ = env.step(action)

                # if self.render:
                # env.render()

                if r == 1:
                    success[context, context_ind[i]] = r
                    # env.close()
                    break

            env.close()

    def test_agent(self, agent, cpu_set=None, epoch=None):
        logging.info("Start testing agent")

        if cpu_set is None:
            num_cpu = mp.cpu_count()
            cpu_set = [i for i in range(num_cpu)]
        else:
            num_cpu = len(cpu_set)

        print("there is {} cpus".format(num_cpu))

        if self.task_suite == "libero_90":
            num_tasks = 90
        else:
            num_tasks = 10

        success = torch.zeros([num_tasks, self.num_episode]).share_memory_()
        all_runs = num_tasks * self.num_episode
        ###################################################################
        # distribute every runs on cpu
        ###################################################################
        contexts = np.arange(num_tasks)
        contexts = np.repeat(contexts, self.num_episode)

        context_ind = np.arange(self.num_episode)
        context_ind = np.tile(context_ind, num_tasks)

        repeat_num = all_runs // num_cpu
        repeat_res = all_runs % num_cpu

        workload_array = np.ones([num_cpu], dtype=int)
        workload_array[:repeat_res] += repeat_num
        workload_array[repeat_res:] = repeat_num

        assert np.sum(workload_array) == all_runs

        ind_workload = np.cumsum(workload_array)
        ind_workload = np.concatenate([[0], ind_workload])
        ###################################################################
        ctx = mp.get_context('spawn')
        processes_list = []

        for i in range(self.n_cores):
            p = ctx.Process(target=self.eval_agent,
                            kwargs={
                                "agent": agent,
                                "contexts": contexts[ind_workload[i]:ind_workload[i + 1]],
                                "context_ind": context_ind[ind_workload[i]:ind_workload[i + 1]],
                                "success": success,
                                "pid": i,
                                "cpu_set": set(cpu_set[i:i + 1])
                            },
                            )
            p.start()
            processes_list.append(p)

        [p.join() for p in processes_list]

        success_rate = torch.mean(success, dim=-1)
        average_success = torch.mean(success_rate).item()

        print(f'success array {success.detach()}')

        custom_step = f"{epoch}_custom_step"
        wandb.define_metric(custom_step)
        wandb.define_metric(f"{epoch}_tasks_success", step_metric=custom_step)

        for num in range(num_tasks):
            log.info(f"Task {num}: {success_rate[num].item()}")

            wandb.log({custom_step: num,
                       f"{epoch}_tasks_success": success_rate[num].item()
                       })

        wandb.log({f"epoch{epoch}_average_success": average_success})
        log.info(f"Average success rate: {average_success}")
