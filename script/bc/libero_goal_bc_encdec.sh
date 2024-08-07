python run_benchmark.py  --config-name=benchmark_libero_goal \
            --multirun agents=goal_bc_encdec \
            agent_name=bc_encdec \
            task_suite=libero_goal,libero_90 \
            wandb.project=3_seed \
            group=bc_encdec_goal \
            obs_seq=1,5 \
            seed=0,1,2