
import time
from tensorflow import keras
from keras import layers
from setup import *
from generators import * 
from utils import *
from full_unet import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
import cv2
from setup import *

training_name = "A"
save_path = './weights_'+training_name
save_log = training_name+".log"

print("starting distributed...")
strategy = tf.distribute.MirroredStrategy()

training = False

print("starting training "+training_name+"...")
image_size = 1072
widths = [64, 128, 256, 512]
block_depth = 3

train_generator = DataGeneratorMemmap("./DATA/sfcWind_CERRA_day_gr55_1985-2010.nc", "./DATA/sfcWind_ERA5_day_gr022_1985-2010.nc","si10","sfcWind",
                              max_cerra_ws, max_era5_ws, min_cerra_ws, min_era5_ws, inshape = 1072,sequential=False, batch_size=2, unet = True)

test_generator = DataGeneratorMemmap("./DATA/sfcWind_CERRA_day_gr55_2011-2020.nc", "./DATA/sfcWind_ERA5_day_gr022_2011-2020.nc","si10","sfcWind",
                              max_cerra_ws, max_era5_ws, min_cerra_ws, min_era5_ws, inshape = 1072,sequential=False, batch_size=2, unet = True)

print("generators initialized...")

with strategy.scope():

    model = get_unet(image_size, 1, 1, widths, block_depth)

    optimizer=keras.optimizers.AdamW
    model.compile(
        optimizer=optimizer(
            learning_rate=1e-4, weight_decay=1e-5
        ),
        loss=tf.keras.losses.MeanSquaredError()
    )

    # Save the model weights with parametric freq. and folder
from keras.callbacks import Callback
class CustomSaveModelCallback(Callback):
    def __init__(self, save_freq = 20, save_path='./'):
        super(CustomSaveModelCallback, self).__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            filename = f'winspeed_unet_{epoch + 1}.weights.h5'
            file_path = os.path.join(self.save_path, filename)
            self.model.save_weights(file_path)
            print(f'Saved weights at epoch {epoch + 1} to {file_path}')
            a, b = test_generator.__getitem__(1)
            c = model.predict(a, verbose = 0)
            plt.imsave(self.save_path + "/gt"+str(epoch)+".png", b[0,:,:,0])
            plt.imsave(self.save_path + "/in"+str(epoch)+".png", a[0,:,:,0])
            plt.imsave(self.save_path + "/out"+str(epoch)+".png", c[0,:,:,0])

save_callback = CustomSaveModelCallback(save_freq = 5, save_path=save_path)


save_callback.on_epoch_end(0)
csv_logger = keras.callbacks.CSVLogger(save_log, append=True)

if(training):

    print("training "+training_name+" condition started")

    #model.load_weights("weights_d/tas_unet_314.weights.h5")
    # run training and plot generated images periodically
    history = model.fit(
        train_generator,
        initial_epoch = 110, 
        validation_data = test_generator, 
        epochs=500,
        steps_per_epoch=500,
        validation_steps=200,
        validation_freq=3,
        verbose = 2,
        #batch_size=32,
        callbacks=[
            save_callback,
            csv_logger
        ],
    )

else:
    print("testing "+training_name+" started")
    model.load_weights(save_path+"/winspeed_unet_225.weights.h5")


    sequential_test_generator = DataGeneratorMemmap("./DATA/sfcWind_CERRA_day_gr55_2011-2020.nc", "./DATA/sfcWind_ERA5_day_gr022_2011-2020.nc","si10","sfcWind",
                                max_cerra_ws, max_era5_ws, min_cerra_ws, min_era5_ws, inshape = 1072,sequential=True, batch_size=2, unet = True)

    import sys 

    def calculate_metrics(true, pred):
        mse_value = batch_mse(np.squeeze(true) , np.squeeze(pred))
        rmse_value = batch_rmse(np.squeeze(true) , np.squeeze(pred))
        mae_value = batch_mae(np.squeeze(true) , np.squeeze(pred))
        ssim_value = batch_ssim(np.squeeze(true), np.squeeze(pred))
        psnr_value = batch_psnr(np.squeeze(true), np.squeeze(pred))
        return mse_value, ssim_value, psnr_value, rmse_value, mae_value

    def experiment_unet(generator, n_iter=100):
        #raw = np.zeros((273,32,256,256))
        #bilinear = np.zeros((91,32,256,256))
        
        mses = np.zeros(n_iter)
        ssims = np.zeros(n_iter)
        psnrs = np.zeros(n_iter)
        maes = np.zeros(n_iter)
        rmses = np.zeros(n_iter)

        mses_baseline = np.zeros(n_iter)
        ssims_baseline = np.zeros(n_iter)
        psnrs_baseline = np.zeros(n_iter)
        rmses_baseline = np.zeros(n_iter)
        maes_baseline = np.zeros(n_iter)

        for i in range(n_iter):
            if(i%20 == 0):
                print(i)

            input_batch, output_batch = generator.__getitem__(i)
            prediction_batch = model.predict(input_batch, verbose=0)

            input_batch_resized = np.zeros((input_batch.shape[0],output_batch.shape[1],output_batch.shape[2],input_batch.shape[3]))
            for l in range(input_batch.shape[0]):
                for m in range(input_batch.shape[3]):
                     input_batch_resized[l,:,:,m] = cv2.resize(input_batch[l,:,:,m],(output_batch.shape[1],output_batch.shape[2]),interpolation=cv2.INTER_LINEAR)
            input_batch = input_batch_resized
            
            mses[i], ssims[i], psnrs[i], rmses[i], maes[i] = calculate_metrics(output_batch, prediction_batch)
            mses_baseline[i], ssims_baseline[i], psnrs_baseline[i], rmses_baseline[i], maes_baseline[i] = calculate_metrics(output_batch, input_batch)
            

            #raw[i] = np.squeeze(tmp)
            #bilinear[i] = sampin[:,:,:,-2]
        # return average of all mses
        return mses, mses_baseline, psnrs, psnrs_baseline, ssims, ssims_baseline, rmses,rmses_baseline,maes,maes_baseline   #, raw, bilinear
    
    import time
    start_time = time.time() 
    mses, mses_baseline, psnrs, psnrs_baseline, ssims, ssims_baseline, rmses,rmses_baseline,maes,maes_baseline = experiment_unet(sequential_test_generator, n_iter=sequential_test_generator.__len__()-1)
    end_time = time.time()  # Capture the end time
    execution_time = end_time - start_time 

    print("Execution time: ",execution_time)

    print(f"ssim : {ssims.mean():.2e}  ssim bilinear : {ssims_baseline.mean():.2e}")
    print(f"psnrs: {psnrs.mean():.2e}  psnrs bilinear: {psnrs_baseline.mean():.2e}")
    print(f"mse  : {mses.mean():.2e}  mse bilinear  : {mses_baseline.mean():.2e}")
    print(f"mae  : {maes.mean():.2e}  mae bilinear  : {maes_baseline.mean():.2e}")
    print(f"rmse : {rmses.mean():.2e}  rmse bilinear : {rmses_baseline.mean():.2e}")
