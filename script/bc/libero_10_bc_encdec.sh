python run_benchmark.py  --config-name=benchmark_libero10 \
            --multirun agents=bc_encdec \
            agent_name=bc_encdec \
            wandb.project=3_seed \
            group=bc_encdec \
            obs_seq=1,5 \
            decoder_n_layer=6 \
            seed=0,1,2