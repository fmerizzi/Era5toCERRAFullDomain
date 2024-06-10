# computed on the whole dataset from 2010-2019 
max_cerra_tas = 317.40024
min_cerra_tas =  204.834
max_era5_tas  =  317.08838
min_era5_tas  =  217.06305

max_cerra_ws =  44.663539
min_cerra_ws =  0.0727703
max_era5_ws  =  33.146915
min_era5_ws  =  0.0916735

max_cerra_pr =  422.844726
min_cerra_pr =  0
max_era5_pr  =  0.00265341408
min_era5_pr  =  0

max_cerra_tas_3h =  324.59597
min_cerra_tas_3h =  209.07814
max_era5_tas_3h  =  323.43365
min_era5_tas_3h  =  214.25631

max_cerra_pr_3h = 189.95703
min_cerra_pr_3h =  0
max_era5_pr_3h  =  863.21893
min_era5_pr_3h  =  0

max_cerra_tas_cut256 = 306.69617
min_cerra_tas_cut256 =  234.27051
max_era5_tas_cut256  =  304.31732
min_era5_tas_cut256  =  235.13172

max_cerra_pr_cut256 = 251.43066
min_cerra_pr_cut256 =  0
max_era5_pr_cut256  =  0.0015520632
min_era5_pr_cut256  =  0

#num_epochs = 200  # train for at least 50 epochs for good results
image_size = 256
num_frames = 4
plot_diffusion_steps = 20

# sampling

min_signal_rate = 0.015
max_signal_rate = 0.95

# architecture

embedding_dims = 64 # 32
embedding_max_frequency = 1000.0
widths = [64, 128, 256, 384]
block_depth = 3

# optimization

#batch_size =  32
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4

