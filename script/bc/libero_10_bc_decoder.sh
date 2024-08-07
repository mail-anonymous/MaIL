python run_benchmark.py  --config-name=benchmark_libero10 \
            --multirun agents=bc_decoder \
            agent_name=bc_decoder \
            wandb.project=3_seed \
            group=bc_decoder \
            obs_seq=1,5 \
            trans_n_layer=8 \
            seed=0,1,2