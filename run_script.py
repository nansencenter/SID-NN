import os
import numpy as np
import autokeras as ak
from sklearn.model_selection import train_test_split


#datadir = '/Data/sim/antonk/shom5km_defor_4cnn'
rootdir = '/mnt/sfe-ns9602k/Julien/data'
datadir = os.path.join(rootdir,'anton/shom5km_defor_4cnn')
traindir = os.path.join(rootdir,'train')

data = np.load(os.path.join(traindir,'train.npz'))

X = data['Xtrain']
y = data['ytrain']
mask_train = data['mask_train']
shape_original = data['shape_original']
print (X.shape[0],mask_train.shape[0])

Xtrain, Xval, ytrain, yval = train_test_split(X, y,
                                                    test_size = 0.15,
                                                    random_state = 1)

input_node = ak.ImageInput()
out_node = ak.Normalization()(input_node)
out_node = ak.ConvBlock()(out_node)
#out_node2 = ak.XceptionBlock()(out_node)
#out_node3 = ak.ResNetBlock()(out_node)
#out_node = ak.Merge()([out_node1, out_node2, out_node3])
output_node = ak.RegressionHead()(out_node)


reg = ak.AutoModel(
    inputs=input_node, outputs=output_node, 
    overwrite=False, 
    max_trials=30,
    project_name = 'reg2'
)

reg.fit(
    Xtrain,
    ytrain,
    validation_data=(Xval, yval),
    epochs=50,
)

model = reg.export_model()
model.save("reg2/model_autokeras", save_format="tf")