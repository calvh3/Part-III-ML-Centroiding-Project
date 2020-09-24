import numpy as np
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
from astropy.io import fits

'''
Load data filenames
'''
specklescreen_names = sorted(glob.glob('/content/specklescreens*.fits'))
Zs_names = sorted(glob.glob('/content/Zs*.fits'))
Gs_names = sorted(glob.glob('/content/Gs*.fits'))


'''
Functions to load and process the data
'''
def load_data(filenames):
  '''Loads data given filenames'''
  N = len(filenames)
  data = []
  for i in filenames:
    data.append(fits.getdata(i,ext=0))
    print('loaded:',i)
  data = np.concatenate(data,axis=0)
  return data

def normalise(images):
'''Normalises images between [0,1]'''
    def normalise_image(image):
        image1 = (image-np.abs(image).min())/(np.abs(image).max()-np.abs(image).min())
        return image1
    images2 = np.zeros(np.shape(images))
    for i in range(np.alen(images)):
        image3 = images[i,:,:]
        image3 = normalise_image(image3)
        images2[i,:,:] = image3
    return images2

def noiser(SpeckleScreen, photon_count):
    '''Simulates shot noise in images'''
    power = np.sum(SpeckleScreen,axis=(1,2))
    SpeckleScreen = SpeckleScreen/power[:,np.newaxis,np.newaxis]*photon_count
    SpeckleScreen = np.random.poisson(np.abs(SpeckleScreen))
    SpeckleScreen = SpeckleScreen*power[:,np.newaxis,np.newaxis]/photon_count
    return SpeckleScreen

def test_train_split(Array,tr_te):
  '''Split data depending on percentage train data (tr_te)'''
  Array_train = Array[0:int(len(Array[:,0])*tr_te)]
  Array_test = Array[int(len(Array[:,0])*tr_te):int(len(Array[:,0]))]
  return Array_train, Array_test
  
  
  
class data:
'''Data class to store keep track of screens, Zs, Gs and functions'''
    def __init__(self,specklescreen_names,Zs_names,Gs_names):
        self.screens = load_data(specklescreen_names)
        self.screens = self.screens[:,:,:,np.newaxis]
        self.Zs = load_data(Zs_names)
        self.Gs = load_data(Gs_names)
        self.Zpredicts = self.Zs-self.Gs
        self.diameter = len(self.screens[0,:,0,0])
  
    def normalise(self):
        self.screens = normalise(self.screens)
    
    def noise(self,photon_count):
        self.screens = noiser(self.screens,photon_count)

    def test_train_split(self,r):
        self.screens_train,self.screens_test = test_train_split(self.screens,r)
        self.Zpredicts_train,self.Zpredicts_test = test_train_split(self.Zpredicts,r)
        
        
 '''Load data'''
speckle = data(specklescreen_names,Zs_names,Gs_names)
speckle.test_train_split(0.9)


'''Define different Neural Networks for testing using Keras'''

def InceptionV3_model(diameter,dropout_rate=0.25, dense_nodes=256):
    '''InceptionV3 model'''
    conv_base = tf.keras.applications.InceptionV3(include_top=False,weights=None,
                        input_shape=(1,diameter, diameter))
    conv_base.trainable = True
    #Define model as layers...
    model = tf.keras.models.Sequential()
    #Add VGG16 network
    model.add(conv_base)
    model.add(tf.keras.layers.Flatten())
    #Dropout
    model.add(tf.keras.layers.Dropout(rate = dropout_rate))
    #Densely connected layer
    model.add(tf.keras.layers.Dense(dense_nodes, activation='relu'))
    #Output layer
    model.add(tf.keras.layers.Dense(2, activation='linear'))
    return model
    
    
from __future__ import division

import six
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    add,
    BatchNormalization
)

from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3



def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, block_fn, repetitions):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)

        model = Model(inputs=input, outputs=flatten1)
        return model

    @staticmethod
    def build_resnet_18(input_shape):
        return ResnetBuilder.build(input_shape, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape):
        return ResnetBuilder.build(input_shape, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape):
        return ResnetBuilder.build(input_shape, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape):
        return ResnetBuilder.build(input_shape, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape):
        return ResnetBuilder.build(input_shape, bottleneck, [3, 8, 36, 3])


def ResNet18_model(diameter,dropout_rate=0.25,dense_nodes=256):
  #Define ResNet base
  conv_base = ResnetBuilder.build_resnet_18((diameter, diameter,1))
  conv_base.trainable = True
  '''
  #Print model summary and save network flow chart diagram...
  print(conv_base.summary())
  from tensorflow.keras.utils import plot_model
  plot_model(conv_base, to_file='/content/drive/My Drive/resnet18.png')
  '''
  #Define model as layers...
  model = tf.keras.models.Sequential()
  #Add ResNet network
  model.add(conv_base)
  #model.add(tf.keras.layers.Flatten())
  #Dropout
  model.add(tf.keras.layers.Dropout(rate = dropout_rate))
  #Densely connected layer
  model.add(tf.keras.layers.Dense(dense_nodes, activation='relu'))
  #Output layer
  model.add(tf.keras.layers.Dense(2, activation='linear'))
  return model

def ResNet50_model(diameter,dropout_rate=0.25,dense_nodes=512):
  #Define ResNet base
  conv_base = ResnetBuilder.build_resnet_50((diameter, diameter,1))
  conv_base.trainable = True
  '''
  #Print model summary and save network flow chart diagram...
  print(conv_base.summary())
  from tensorflow.keras.utils import plot_model
  plot_model(conv_base, to_file='/content/drive/My Drive/resnet18.png')
  '''
  #Define model as layers...
  model = tf.keras.models.Sequential()
  #Add ResNet network
  model.add(conv_base)
  #model.add(tf.keras.layers.Flatten())
  #Dropout
  model.add(tf.keras.layers.Dropout(rate = dropout_rate))
  #Densely connected layer
  model.add(tf.keras.layers.Dense(dense_nodes, activation='relu'))
  #Output layer
  model.add(tf.keras.layers.Dense(2, activation='linear'))
  return model


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout

def custom_VGG_16_base(diameter):
    model = tf.keras.models.Sequential()
    input_shape = (diameter,diameter,1)
    img_input = tf.keras.layers.Input(shape=input_shape)
    # Block 1
    x = Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
    #x = Dropout(0.1)(x)

    # Block 2
    x = Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    #x = Dropout(0.05)(x)
    
    # Block 3
    x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Dropout(0.05)(x)
    
    # Block 4
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Dropout(0.1)(x)
    
    # Block 5
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # Create model.
    model = tf.keras.models.Model(img_input, x, name='vgg16_custom')
    print(model.summary())
    return model
  
def custom_VGG16_model(diameter,dropout_rate=0.2, dense_nodes=512):
  '''Vgg16 model'''
  #Use Keras vgg16 model base
  conv_base = custom_VGG_16_base(diameter)
  conv_base.trainable = True

  #Define model as layers...
  model = tf.keras.models.Sequential()
  #Add VGG16 network
  model.add(conv_base)
  model.add(tf.keras.layers.Flatten())
  #Dropout
  model.add(tf.keras.layers.Dropout(rate = dropout_rate))
  #Densely connected layer
  model.add(tf.keras.layers.Dense(dense_nodes, activation='relu'))
  #Output layer
  model.add(tf.keras.layers.Dense(2, activation='linear'))
  return model
  
  
'''Choose network'''
  
model_name = 'custom_vgg16'

elif model_name == 'Vgg16':
  model = Vgg16_model(speckle.diameter)
elif model_name == 'InceptionV3':
  model = InceptionV3_model(speckle.diameter)
elif model_name == 'custom_vgg16':
  model = custom_VGG16_model(speckle.diameter)
elif model_name == 'ResNet50':
  model = ResNet50_model(speckle.diameter)
elif model_name == 'ResNet18':
  model = ResNet18_model(speckle.diameter)

print(model.summary())


'''optionally load a previous model'''
#model.load_weights('/content/checkpoints.hdf5')

'''Train model'''
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(lr=1e-5))
              #optimizer=tf.keras.optimizers.SGD(lr= 2e-4,momentum = 0.9))
              #optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9))

filepath1='/content/drive/My Drive/ML to Centroid Speckle Images/Training Data/Fixed_r0/Architecture_Testing/64x64_Dr0_5/downsample12/{}_v1_checkpoints.hdf5'.format(model_name)
filepath2='/content/drive/My Drive/ML to Centroid Speckle Images/Training Data/Fixed_r0/Architecture_Testing/64x64_Dr0_5/downsample12/{}_v1_trainingdata.csv'.format(model_name)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath1, monitor='val_loss', verbose=1, save_best_only=True)
trainingdata = tf.keras.callbacks.CSVLogger(filepath2, separator=',', append=False)

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)

callbacks_list = [checkpoint,trainingdata,earlystopping]

history = model.fit(speckle.screens_train, speckle.Zpredicts_train,
          batch_size=64,
          epochs=80,
          verbose=1,
          validation_data=(speckle.screens_test, speckle.Zpredicts_test),
          callbacks=callbacks_list)
          
          
          
'''Plot training data''''

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
