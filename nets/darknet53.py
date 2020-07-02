from functools import wraps
import keras
from keras.layers import Conv2D, Add, Lambda, Activation, Multiply, ZeroPadding2D, Concatenate, Permute, Input, Dense, GlobalAveragePooling2D, Reshape, GlobalMaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from utils.utils import compose


#--------------------------------------------------#
#   单次卷积
#--------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

#---------------------------------------------------#
#   CBAM模块
#   channel attention + spectial attention
#---------------------------------------------------#

def cbam_block(input_feature, ratio=8):
    kernel_size = 7
    channel_axis = 1 if keras.backend.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]
    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    spectial_result = Activation('sigmoid')(cbam_feature)

    channel_result = Multiply()([input_feature, spectial_result])

    avg_pool = Lambda(lambda x: keras.backend.mean(x, axis=3, keepdims=True))(channel_result)
    max_pool = Lambda(lambda x: keras.backend.max(x, axis=3, keepdims=True))(channel_result)

    concat = Concatenate(axis=3)([avg_pool, max_pool])

    channel_result = Conv2D(filters=1,
                            kernel_size=kernel_size,
                            strides=1,
                            padding='same',
                            activation='sigmoid',
                            kernel_initializer='he_normal',
                            use_bias=False)(concat)

    if keras.backend.image_data_format() == "channels_first":
        channel_result = Permute((3, 1, 2))(channel_result)

    return Multiply()([input_feature, channel_result])

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#

def resblock_body(x, num_filters, num_blocks):
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(num_filters//2, (1,1))(x)
        y = DarknetConv2D_BN_Leaky(num_filters, (3,3))(y)
        x = Add()([x, y])
    return cbam_block(x)

#---------------------------------------------------#
#   darknet53 的主体部分
#---------------------------------------------------#
def darknet_body(x):
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    feat1 = x
    x = resblock_body(x, 512, 8)
    feat2 = x
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1,feat2,feat3

#测试模型是否正常
#x=Input(shape=(None,None,3))
#y1,y2,y3=darknet_body(x)
#model=keras.models.Model(input=(x),output=[y1,y2,y3])
#print(model.summary())
