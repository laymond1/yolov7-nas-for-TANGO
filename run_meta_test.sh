python main.py\
	--mode meta-test\
	--seed 42\
	--meta_lr 5e-5\
	--layer_size 250\
	--meta_train_devices 'galaxy_s10_cpu_head,galaxy_s10_gpu_head,galaxy_s22_nnapi_head'\
	--meta_valid_devices 'galaxy_s22_cpu_head,galaxy_s22_gpu_head'\
	--meta_test_devices 'galaxy_s10_nnapi_head'