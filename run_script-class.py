import os
import numpy as np
import autokeras as ak
from sklearn.model_selection import train_test_split
import yaml 
import tensorflow as tf
from dti_util import code_dam, decode_dam

# Name of the experiment
name = 'exp4'

# Root of all directory used
rootdir = '/mnt/sfe-ns9602k/Julien/data'

# Directory of the experiment outputs
expdir = os.path.join(rootdir, name)

# Load experiment parameters
with open(os.path.join(expdir,'data_params.yml' )) as file:
    exp_dict = yaml.load(file, Loader=yaml.FullLoader)
    
# Print experiments parameters
for key, value in exp_dict.items():
    print(key, ' : ', value)
    
# Set the used parameters
traindir = exp_dict['traindir']
epsi = exp_dict['epsi']
th_dam = exp_dict['th_dam']
dsize = exp_dict['dsize']


# Name of the model
#model_name = "long2"
model_name = "long-class"

# Directory to stores the model
model_dir = os.path.join(expdir, model_name)

# Create the model directory if necessary
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

# Read model parameters from yml file
with open(os.path.join(model_dir,'model_params.yml' )) as file:
    dmod = yaml.load(file, Loader=yaml.FullLoader)        

print ('--- MODEL CONFIGURATION ---')
for key, value in dmod.items():
    print(key, ' : ', value)

# Normalization function
norm = lambda x : code_dam(x,epsi=epsi, vmin=th_dam)

# Denormalization function
denorm = lambda x : decode_dam(x,epsi=epsi, vmin=th_dam)

data = np.load(os.path.join(traindir,'train.npz'))
X = data['Xtrain']
y = data['ytrain']
mask_train = data['mask_train']
yd = denorm(y)

print(f"Number of training samples: {X.shape[0]}")
print(f'Size of input feature: {X.shape[1:]}')

Xtrain, Xval, ytrain, yval = train_test_split(X.astype(np.float32), yd.astype(np.float32),
                                                    test_size = dmod['test_size'],
                                                    random_state = dmod['split_seed'])

print(f"Number of training samples: {X.shape[0]}")
print(f"Number of validation samples: {X.shape[0]}")
ytrain_c = ytrain > dmod['target_th']
yval_c = yval > dmod['target_th']
c_th = dmod['target_th']

print(f"Number of damage in val <= {c_th}: {np.sum(~yval_c)}")
print(f"Number of damage in val > {c_th}: {np.sum(yval_c)}")

input_node = ak.ImageInput()
out_node = ak.Normalization()(input_node)
out_node = ak.ConvBlock()(out_node)
#out_node2 = ak.XceptionBlock()(out_node)
#out_node3 = ak.ResNetBlock()(out_node)
#out_node = ak.Merge()([out_node1, out_node2, out_node3])
output_node = ak.ClassificationHead()(out_node)


reg  = ak.AutoModel(
    inputs=input_node, outputs=output_node, 
    overwrite=True, 
    max_trials=dmod['max_trials'],
    project_name = 'reg',
    directory=model_dir
)


tensorboard_callback= tf.keras.callbacks.TensorBoard(log_dir=dmod['log_dir'], histogram_freq=2)
early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=dmod['patience'],
    min_delta=1e-4,
    verbose=1,
    restore_best_weights=True
    )

reg.fit(
    Xtrain[:dmod['ntrain']],
    ytrain_c[:dmod['ntrain']],
    validation_data=(Xval, yval_c),
    epochs = dmod['epochs'],
    callbacks=[tensorboard_callback, early_stopping_monitor],
    #batch_size=16
    )
model = reg.export_model()
model.save(os.path.join(model_dir,model_name), save_format="tf")

model.summary()