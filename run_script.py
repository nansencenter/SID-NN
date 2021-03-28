import os
import numpy as np
import autokeras as ak
from sklearn.model_selection import train_test_split
import yaml 
import tensorflow as tf
# Name of the experiment
name = 'demo'

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
model_name = "long"

# relative size of the validation dataset
test_size = 0.15

# Seed of the train/val random split
split_seed = 1


# Directory to stores the model
model_dir = os.path.join(expdir, model_name)

# Normalization function
norm = lambda x : code_dam(x,epsi=epsi, vmin=th_dam)

# Denormalization function
denorm = lambda x : decode_dam(x,epsi=epsi, vmin=th_dam)

data = np.load(os.path.join(traindir,'train.npz'))
X = data['Xtrain']
y = data['ytrain']
mask_train = data['mask_train']

print(f"Number of training samples: {X.shape[0]}")
print(f'Size of input feature: {X.shape[1:]}')

Xtrain, Xval, ytrain, yval = train_test_split(X, y,
                                                    test_size = test_size,
                                                    random_state = split_seed)

print(f"Number of training samples: {X.shape[0]}")
print(f"Number of validation samples: {X.shape[0]}")

input_node = ak.ImageInput()
out_node = ak.Normalization()(input_node)
out_node = ak.ConvBlock()(out_node)
#out_node2 = ak.XceptionBlock()(out_node)
#out_node3 = ak.ResNetBlock()(out_node)
#out_node = ak.Merge()([out_node1, out_node2, out_node3])
output_node = ak.RegressionHead()(out_node)


output_node = ak.RegressionHead()(out_node)
reg  = ak.AutoModel(
    inputs=input_node, outputs=output_node, 
    overwrite=True, 
    max_trials=50,
    project_name = 'reg',
    directory=model_dir
)

log_dir=  "/mnt/sfe-ns9602k/.tools/deep-learn-1603280065-tensorboard/autokeras/{}-{}".format(name, model_name)
tensorboard_callback= tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=2)

reg.fit(
    Xtrain,
    ytrain,
    validation_data=(Xval, yval),
    epochs = 50,
    callbacks=[tensorboard_callback]
)
model = reg.export_model()
model.save(os.path.join(model_dir,model_name), save_format="tf")

model.summary()