
import time
from tensorflow import keras
from keras import layers
from setup import *
from generators import * 
from utils import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
import cv2
from setup import *
import os

#os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

training_name = "S"
save_path = './weights_'+training_name
save_log = training_name+".log"

print("starting distributed...")
strategy = tf.distribute.MirroredStrategy()

training = True

print("starting file "+training_name+"...")

output_frames = 1
num_epochs = 10  # For real training, use num_epochs=100. 10 is a test value
image_size = 1072  # We'll resize input images to this size
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 512
num_heads = 6
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 4
mlp_head_units = [
    512,
    256,
]  # Size of the dense layers of the final classifier
input_shape = (image_size, image_size, 1)


train_generator = DataGeneratorMemmap("../sfcWind_CERRA_3h_gr55_1985-2010.nc", "../sfcWind_ERA5_3h_gr022_1985-2010.nc","sfcWind","sfcWind",
                              max_cerra_ws_3h, max_era5_ws_3h, min_cerra_ws_3h, min_era5_ws_3h, inshape = 1072,sequential=False, batch_size=3, unet = True)

test_generator = DataGeneratorMemmap("../sfcWind_CERRA_3h_gr55_2011-2020.nc", "../sfcWind_ERA5_3h_gr022_2011-2020.nc","sfcWind","sfcWind",
                              max_cerra_ws_3h, max_era5_ws_3h, min_cerra_ws_3h, min_era5_ws_3h, inshape = 1072,sequential=False, batch_size=3, unet = True)


print("generators initialized...")

def savefig(img_a,img_b,img_c,save_path, epoch):
    fig, axes = plt.subplots(1, 3, figsize=(40, 20))
    # Plot each image
    axes[0].imshow(img_a)
    axes[0].axis('off')
    axes[0].set_title('ERA5')

    axes[1].imshow(img_b)
    axes[1].axis('off')
    axes[1].set_title('CERRA')

    axes[2].imshow(img_c)
    axes[2].axis('off')
    axes[2].set_title('U-Net')

    # Save the figure
    plt.savefig(save_path + "/wsepoch_{}.png".format(epoch), bbox_inches='tight')
    plt.close()

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = tf.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = tf.image.extract_patches(images, sizes=[1,self.patch_size,self.patch_size,1], 
                                           strides = [1,self.patch_size,self.patch_size,1],
                                           rates = [1,1,1,1], padding = "VALID")
        patches = tf.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.expand_dims(
            tf.range(0, self.num_patches, 1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config


# only used in postprocessing 
def ResidualBlock(width):
            def apply(x):
                input_width = x.shape[3]
                if input_width == width:
                    residual = x
                else:
                    residual = layers.Conv2D(width, kernel_size=1)(x)
                #x = layers.BatchNormalization(center=False, scale=False)(x)
                x = layers.LayerNormalization(axis=-1,center=True, scale=True)(x)
                x = layers.Conv2D(
                    width, kernel_size=3, padding="same", activation=keras.activations.swish
                )(x)
                x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
                x = layers.Add()([x, residual])
                return x
        
            return apply



def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def create_vit():
    inputs = keras.Input(shape=input_shape)
    # Create patches
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    out = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Spatial reshape of the patches
    out = layers.Reshape((67, 67, 512))(out)
    # Residual block
    out = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation="swish")(out)
    out = ResidualBlock(512)(out)
    # Upsample
    
    out = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation="swish")(out)
    out = ResidualBlock(256)(out)
    # Upsample while reducing channel size
    out = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation="swish")(out)
    out = ResidualBlock(128)(out)
    out = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation="swish")(out)
    out = ResidualBlock(64)(out)
    # out = layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation='relu')(out)
    out = layers.Conv2D(output_frames, kernel_size=1, kernel_initializer="zeros")(out)
    out = layers.Conv2D(1, 3, padding="valid")(out)
    out = layers.Conv2D(1, 2, padding="valid")(out)

    model = keras.Model(inputs=inputs, outputs=out)
    return model

with strategy.scope():

    model = create_vit()

    optimizer = keras.optimizers.AdamW(
        learning_rate=1e-4, weight_decay=1e-5
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
    )



print(model.summary())
    # Save the model weights with parametric freq. and folder
from keras.callbacks import Callback
class CustomSaveModelCallback(Callback):
    def __init__(self, save_freq = 20, save_path='./'):
        super(CustomSaveModelCallback, self).__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        a, b = test_generator.__getitem__(1)
        c = model.predict(a, verbose = 0)
        savefig(a[0,:,:,0],b[0,:,:,0],c[0,:,:,0],self.save_path,epoch)
        if (epoch + 1) % self.save_freq == 0:
            filename = f'tas_vit_{epoch + 1}.weights.h5'
            file_path = os.path.join(self.save_path, filename)
            self.model.save_weights(file_path)
            print(f'Saved weights at epoch {epoch + 1} to {file_path}')

save_callback = CustomSaveModelCallback(save_freq = 5, save_path=save_path)


save_callback.on_epoch_end(0)
csv_logger = keras.callbacks.CSVLogger(save_log, append=True)

if(training):

    print("training "+training_name+" condition started")

    #model.load_weights(save_path+"/tas_vit_340.weights.h5")
    # run training and plot generated images periodically
    history = model.fit(
        train_generator,
        initial_epoch = 0, 
        validation_data = test_generator, 
        epochs=400,
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
    model.load_weights(save_path+"/tas_vit_450.weights.h5")
    
    sequential_test_generator = DataGeneratorMemmap("../tas_CERRA_3h_gr55_2011-2020.nc", "../tas_ERA5_3h_gr022_2011-2020.nc","t2m","t2m", 
                              max_cerra_tas_3h, max_era5_tas_3h, min_cerra_tas_3h, min_era5_tas_3h, inshape = 1072,sequential=False, batch_size=8, unet = True)


    #raise Exception("Stopping the program")
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
    mses, mses_baseline, psnrs, psnrs_baseline, ssims, ssims_baseline, rmses,rmses_baseline,maes,maes_baseline = experiment_unet(sequential_test_generator, n_iter=600)
    end_time = time.time()  # Capture the end time
    execution_time = end_time - start_time 

    print("Execution time: ",execution_time)

    print(f"ssim : {ssims.mean():.2e}  ssim bilinear : {ssims_baseline.mean():.2e}")
    print(f"psnrs: {psnrs.mean():.2e}  psnrs bilinear: {psnrs_baseline.mean():.2e}")
    print(f"mse  : {mses.mean():.2e}  mse bilinear  : {mses_baseline.mean():.2e}")
    print(f"mae  : {maes.mean():.2e}  mae bilinear  : {maes_baseline.mean():.2e}")
    print(f"rmse : {rmses.mean():.2e}  rmse bilinear : {rmses_baseline.mean():.2e}")
