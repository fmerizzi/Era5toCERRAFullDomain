
from generators import * 
import numpy as np
import sys


print("calc started")

#train_generator = DataGeneratorMemmap("../tas_CERRA_3h_gr55_2011-2020.nc", "../tas_ERA5_3h_gr022_2011-2020.nc","t2m","t2m", 
#                              1, 1, 0, 0, inshape = 1072,sequential=True, batch_size=2, unet = True)

train_generator = DataGeneratorMemmap("./DATA/pr_CERRA_day_1985-2010.nc", "./DATA/pr_ERA5_day_1985-2010.nc","tp","pr", 
                              1, 1, 0, 0, inshape = 1072,sequential=True, batch_size=2, unet = True)

max_cerra = 0
min_cerra = 400
max_era = 0
min_era = 400

#train_generator.__len__()
len = train_generator.__len__()-1

for i in range(len):
    sys.stdout.write(f'\rProgress: {((i / len) * 100):.2f}%')
    sys.stdout.flush()
    a,b = train_generator.__getitem__(1)
    if(a.max() > max_era):
        max_era = a.max()
    if(a.min() < min_era):
        min_era = a.min()
    if(b.max() > max_cerra):
        max_cerra = b.max()
    if(b.min() < min_cerra):
        min_cerra = b.min()

print(" ")
print(max_cerra)
print(min_cerra)
print(max_era)
print(min_era)
