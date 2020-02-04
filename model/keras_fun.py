from keras.layers import *
import keras.backend as K
import tensorflow as tf
from keras_applications.mobilenet import relu6
class BilinearUpsampling(Layer):
    def __init__(self, upsampling=(2, 2), data_format=None, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'size')
        self.input_spec = InputSpec(ndim=4)
    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
            input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
            input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])
    def call(self, inputs):
        return K.tf.image.resize_bilinear(inputs, (int(inputs.shape[1]*self.upsampling[0]),
                                                   int(inputs.shape[2]*self.upsampling[1])))
    def get_config(self):
        config = {'size': self.upsampling,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def aspp(x,input_shape,out_stride):
    b0=Conv2D(128,(1,1),kernel_initializer='he_normal',padding="same",use_bias=True)(x)
    b0=BatchNormalization()(b0)
    b0=Activation("relu")(b0)

    b1=Conv2D(128,(3,3),kernel_initializer='he_normal',dilation_rate=(6,6),padding="same",use_bias=True)(x)
    b1=BatchNormalization()(b1)
    b1=Activation("relu")(b1)

    b2=Conv2D(128,(3,3),kernel_initializer='he_normal',dilation_rate=(12,12),padding="same",use_bias=True)(x)
    b2=BatchNormalization()(b2)
    b2=Activation("relu")(b2)

    b3=Conv2D(128,(3,3),kernel_initializer='he_normal',dilation_rate=(18,18),padding="same",use_bias=True)(x)
    b3=BatchNormalization()(b3)
    b3=Activation("relu")(b3)

    out_shape=int(input_shape[0]/out_stride)
    b4=MaxPooling2D(pool_size=(out_shape,out_shape))(x)
    b4=Conv2D(128,(1,1),kernel_initializer='he_normal',padding="same",use_bias=True)(b4)
    b4=BatchNormalization()(b4)
    b4=Activation("relu")(b4)
    b4=BilinearUpsampling((out_shape,out_shape))(b4)

    x=Concatenate()([b4,b0,b1,b2,b3])
    return x

def aspp_plus(x,input_shape,out_stride):
    b0=Conv2D(256,(1,1),padding="same",use_bias=True)(x)
    b0=BatchNormalization()(b0)
    b0=Activation(relu6)(b0)

    b1=DepthwiseConv2D((3,3),dilation_rate=(6,6),padding="same",use_bias=True)(x)
    b1=BatchNormalization()(b1)
    b1=Activation(relu6)(b1)
    b1=Conv2D(256,(1,1),padding="same",use_bias=False)(b1)
    b1=BatchNormalization()(b1)
    b1=Activation(relu6)(b1)

    b2=DepthwiseConv2D((3,3),dilation_rate=(12,12),padding="same",use_bias=True)(x)
    b2=BatchNormalization()(b2)
    b2=Activation(relu6)(b2)
    b2=Conv2D(256,(1,1),padding="same",use_bias=True)(b2)
    b2=BatchNormalization()(b2)
    b2=Activation(relu6)(b2)

    b3=DepthwiseConv2D((3,3),dilation_rate=(18,18),padding="same",use_bias=True)(x)
    b3=BatchNormalization()(b3)
    b3=Activation(relu6)(b3)
    b3=Conv2D(256,(1,1),padding="same",use_bias=True)(b3)
    b3=BatchNormalization()(b3)
    b3=Activation(relu6)(b3)

    out_shape=int(input_shape[0]/out_stride)
    b4=AveragePooling2D(pool_size=(out_shape,out_shape))(x)
    b4=Conv2D(256,(1,1),padding="same",use_bias=True)(b4)
    b4=BatchNormalization()(b4)
    b4=Activation(relu6)(b4)
    b4=BilinearUpsampling((out_shape,out_shape))(b4)

    x=Concatenate()([b4,b0,b1,b2,b3])
    return x

def _make_divisible(v,divisor,min_value=None):
    if min_value is None :
        min_value =divisor
    new_v = max(min_value,int(v+divisor/2)// divisor*divisor)
    if new_v < 0.9*v :
        new_v += divisor
    return new_v

def _conv_block(inputs, filters, kernel, strides):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x =  Conv2D(filters,kernel,padding="same",strides=strides)(inputs)
    return Activation(relu6)(x)

def _bottleneck(inputs,filters,kernel,t,alpha,s,r=False):
    channel_axis = 1 if K.image_data_format() =='channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis]*t
    cchannel = int(filters * alpha)
    x = _conv_block(inputs,tchannel,(1,1),(1,1))
    x = DepthwiseConv2D(kernel,strides=(s,s),depth_multiplier=1,padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)
    x = Conv2D(cchannel, (1,1), strides=(1,1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    if r :
        x = add([x,inputs])
    return x

def _inverted_residual_block(inputs,filters,kernel,t,alpha,strides,n):
    x = _bottleneck(inputs,filters,kernel,t,alpha,strides)
    for  i in range(1,n):
        x = _bottleneck(x,filters,kernel,t,alpha,1,True)
    return x