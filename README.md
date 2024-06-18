# Era5toCERRAFullDomain
A repository containing the code for downscaling from ERA5 to CERRA with multiple neural architectures on the full domain

- setup.py, global variables definition
- utils.py, accessory functions, metrics
- generators.py, code to generate batches of data starting from nc files
- full_unet.py, architectural definition of the U-net for full domain
- plot_log.py, accessory function that plot training history starting from the log files produced while training
- computation_max_min.py, compute maximum and minimum for the nc dataset, normalization purposes
- UNET_ws_daily_A.py, training + testing file for daily wind speed
- UNET_pr_3h_full_sizeF.py, training + testing file for 3h precipitation
- UNET_pr_daily_B.py, training + testing file for daily precipitation
- UNET_pr_daily_spatt_C.py, training + testing file for daily precipitation with spatial attention
- full_spatial_unet.py, architectural definition of the U-net for full domain with spatial attention
- UNET_tas_3h_full_sizeE.py, training + testing file for 3h temperature
- UNET_tas_daily_full_sizeD.py, training + testing file for daily temperature
  
