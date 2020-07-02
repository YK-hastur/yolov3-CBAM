from functools import wraps
import keras.backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate
from keras.layers import Input, GlobalAveragePooling2D, Reshape, Dense, Permute, multiply, Activation, add, Lambda, concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from utils.utils import compose


@wraps(Conv2D)
def pDarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def pDarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        pDarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def presblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = pDarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            pDarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            pDarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        y = squeeze_excite_block(y)
        x = Add()([x, y])
    return x


# https://github.com/titu1994/keras-squeeze-excite-network/blob/master/keras_squeeze_excite_network/se_resnet.py
def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, kernel_initializer='he_normal', use_bias=False)(se)
    se = LeakyReLU(alpha=0.1)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def _tensor_shape(tensor):
    return getattr(tensor, '_keras_shape')



def pdarknet_body(x):
    base = 4  # orig base = 8
    x = pDarknetConv2D_BN_Leaky(base * 4, (3, 3))(x)
    x = presblock_body(x, base * 8, 1)
    x = presblock_body(x, base * 16, 2)
    x = presblock_body(x, base * 32, 8)
    small = x
    x = presblock_body(x, base * 64, 8)
    medium = x
    x = presblock_body(x, base * 128, 8)
    big = x
    return small, medium, big



def pmake_last_layers(x, num_filters, out_filters):
    x = compose(
        pDarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        pDarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        pDarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        pDarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        pDarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
        pDarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        pDarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def pyolo_body(inputs, num_anchors, num_classes):
    """Create Poly-YOLO model CNN body in Keras."""
    small, medium, big = pdarknet_body(inputs)

    base = 4
    small  = pDarknetConv2D_BN_Leaky(base * 32, (1, 1))(small)
    medium = pDarknetConv2D_BN_Leaky(base * 32, (1, 1))(medium)
    big    = pDarknetConv2D_BN_Leaky(base * 32, (1, 1))(big)

    all = Add()([medium, UpSampling2D(2,interpolation='bilinear')(big)])
    all = Add()([small, UpSampling2D(2,interpolation='bilinear')(all)])



    num_filters = base*32

    x = compose(
        pDarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        pDarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        pDarknetConv2D_BN_Leaky(num_filters, (1, 1)))(all)

    all = compose(
        pDarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        pDarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))(x)

    return Model(inputs, all)