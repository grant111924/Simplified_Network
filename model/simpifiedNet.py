from keras.models import Model
from keras.layers import Input,Conv2D,GlobalAveragePooling2D,DepthwiseConv2D,UpSampling2D,AveragePooling2D,Conv2DTranspose
from keras.layers import Activation,BatchNormalization,add,average,Reshape,Concatenate,Dropout
from keras_applications.mobilenet import relu6
from keras.utils import plot_model
from keras import backend as K
from model.keras_fun import _make_divisible,_conv_block,_inverted_residual_block
from model.keras_fun import aspp,BilinearUpsampling,aspp_plus

def GoodNet2(size_set=256,alpha=1.0):
    inputs = Input(shape=(size_set,size_set,3))
    inputs2 = AveragePooling2D(pool_size=(2, 2))(inputs)
    inputs2 = _conv_block(inputs2, 16,(3,3), strides=(1,1))

    first_filters=_make_divisible(32*alpha,8)

    x = _conv_block(inputs, first_filters,(3,3), strides=(2,2))

    # block1 = _inverted_residual_block(x,32,(3,3),t=1,strides=2,n=1) #128
    block_concat1 = Concatenate()([x,inputs2])

    block2 = _inverted_residual_block(block_concat1,64,(3,3),alpha=alpha,t=6,strides=2,n=2) #64
    block3 = _inverted_residual_block(block2,96,(3,3),t=6,alpha=alpha,strides=2,n=3) #32
    block4 = _inverted_residual_block(block3,160,(3,3),t=6,alpha=alpha,strides=2,n=4) #16
    block5 = _inverted_residual_block(block4,320,(3,3),t=6,alpha=alpha,strides=1,n=1) #16

    if alpha >1.0:
        last_filters =_make_divisible(1280*alpha,8)
    else :
        last_filters =1280

    block_final = _conv_block(block5,last_filters,(1,1),strides=(1,1))
    aspp = aspp_plus(block_final,input_shape=(size_set,size_set,3),out_stride=16) #16
    # up1 = BilinearUpsampling((4,4))(aspp)
    up1 = Conv2DTranspose(640, (2, 2), strides=(4, 4), padding='same', name='dconv1')(aspp)
    low =_conv_block(block2,128,(1,1),strides=(1,1))

    conat2 = Concatenate()([low,up1])
    conat2 = _conv_block(conat2,128,(3,3),strides=(1,1))

    # dconv =Conv2DTranspose(96, (2, 2), strides=(2, 2), padding='same', name='dconv1')(conat2)
    # dconv =Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='dconv2')(dconv)
    # conv3 = _conv_block(dconv,16,(3,3),strides=(1,1))
    up2 = BilinearUpsampling((2,2))(conat2)
    up2 = _conv_block(up2,32,(3,3),strides=(1,1))
    up3 = BilinearUpsampling((2,2))(up2)
    conv3 = _conv_block(up3,16,(3,3),strides=(1,1))


    out_OD_dice_2=Conv2D(1 , (1, 1), activation='sigmoid', name='out_OD_dice_2')(conv3)
    out_OC_dice_2=Conv2D(1 , (1, 1), activation='sigmoid', name='out_OC_dice_2')(conv3)
    out_OCD_dice_2= Concatenate()([out_OC_dice_2,out_OD_dice_2])
    out_OCD_dice_2= _conv_block(out_OCD_dice_2,16,(3,3),strides=(1,1))
    out_OCD_dice_2= Conv2D(2 , (1, 1), activation='sigmoid', name='out_OCD_dice_2')(out_OCD_dice_2)
    model = Model(inputs,[out_OD_dice_2,out_OC_dice_2,out_OCD_dice_2])
    model.summary()
    # plot_model(model,to_file='MobileNetV2.png',show_shapes=True)

    return model


# if __name__ == '__main__':
#     GoodNet2(400)
