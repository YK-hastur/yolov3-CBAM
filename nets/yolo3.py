from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from nets.darknet53 import darknet_body
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
#   特征层->最后的输出
#---------------------------------------------------#
def make_last_layers(x, num_filters, out_filters):
    # 五次卷积
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)

    # 将最后的通道数调整为outfilter
    y = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    y = DarknetConv2D(out_filters, (1,1))(y)
            
    return x, y

#---------------------------------------------------#
#   特征层->最后的输出
#---------------------------------------------------#
def yolo_body(inputs, num_anchors, num_classes):
    # 生成darknet53的主干模型
    feat1,feat2,feat3 = darknet_body(inputs)
    darknet = Model(inputs, feat3)

    # 第一个特征层
    # y1=(batch_size,13,13,3,85)
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,feat2])
    # 第二个特征层
    # y2=(batch_size,26,26,3,85)
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,feat1])
    # 第三个特征层
    # y3=(batch_size,52,52,3,85)
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])

def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])

#---------------------------------------------------#
#   将预测值的每个特征层调成真实值
#---------------------------------------------------#
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    #feats-->yolo_outputs[?]，shape-->(N,13,13,3*?)、(N,26,26,3*?)、(N,52,52,3*?)
    # input_shape =（416,416）
    num_anchors = len(anchors)
    # [1, 1, 1, num_anchors, 2]
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # 获得x，y的网格
    # (13, 13, 1, 2)
    # grid_shape-->(13,13)、(26,26)、(52,52)
    grid_shape = K.shape(feats)[1:3] # height, width
    #K.tile-->按照channel数列（[1, grid_shape[1], 1, 1]）每个位置对应的维度复制n倍
    #K.arange-->生成一个整数数列
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats)) #K.cast，转换数据格式
    #grid-->[[0,0],[0,1],[0,2]...[1,0],[1,1],[1,2]...]

    # (batch_size,13,13,3,85)
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # 将预测值调成真实值
    # box_xy对应整幅图像的中心点
    # box_wh对应整幅图像的宽和高
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats)) #grid为偏移 ，将x,y相对于featuremap尺寸进行了归一化,将xy范围转换为0-1之间的数值，成为图片中长宽的比例
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats)) #同上，将anchor框缩小为物体的大小，然后进行归一化
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # 在计算loss的时候返回如下参数
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

#---------------------------------------------------#
#   对box进行调整，使其符合真实图片的样子
#---------------------------------------------------#
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape): # 将box_xy, box_wh转换为输入图片上的真实坐标，输出boxes是框的左下、右上两个坐标(y_min, x_min, y_max, x_max)
    #为方便处理
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    #转换类型-->float32，方便指数运算
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    #将预处理后图片得出的位置和长宽比转换为原图片的位置和长宽比
    #input_shape-->(416,416)
    new_shape = K.round(image_shape * K.min(input_shape/image_shape)) #将原图片转化为预处理图片后，图片的尺寸
    offset = (input_shape-new_shape)/2./input_shape #计算偏移量
    scale = input_shape/new_shape #计算缩放量

    box_yx = (box_yx - offset) * scale #将比例坐标移除偏移和缩放
    box_hw *= scale #将长宽缩放
    # 通过中心点坐标计算左上角（min）和右下角（max）在图片中的坐标
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    boxes *= K.concatenate([image_shape, image_shape]) #将比例坐标转化为原图片的真实坐标
    return boxes #原图像的真实坐标[y_min,x_min,y_max,x_max]

#---------------------------------------------------#
#   获取每个box和它的得分
#---------------------------------------------------#
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    # 将预测值调成真实值
    # box_xy对应框的中心点
    # box_wh对应框的宽和高
    # input_shape =（416,416）
    # -1,13,13,3,2; -1,13,13,3,2; -1,13,13,3,1; -1,13,13,3,80
    #box_xy：xy坐标，box_wh：长宽，box_confidence：框的置信度，box_class_probs：框内物体类别的概率，所有参数均对应原图片的真实位置
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    # 将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    # 获得得分和box
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs #[-1,13,13,3,80]
    box_scores = K.reshape(box_scores, [-1, num_classes]) #[-1,80]
    return boxes, box_scores

#---------------------------------------------------#
#   图片预测
#---------------------------------------------------#
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """
    Evaluate YOLO model on given input and return filtered boxes.
    """

    """      
    yolo_outputs        #模型输出，格式如下[（?，13,13,255）（?，26,26,255）（?,52,52,255）] ?:bitch size; 13、26、52:多尺度预测； 255：预测值（3*（80+5））
    anchors,            #[(10,13)，(16,30), (33,23), (30,61), (62,45), (59,119), (116,90), (156,198),(373,326)]
    num_classes,        #类别个数，coco集80类
    image_shape,        #placeholder类型的TF参数，默认(416, 416)；
    max_boxes=20,       #每张图每类最多检测到20个框同类别框的IoU阈值，大于阈值的重叠框被删除，重叠物体较多，则调高阈值，重叠物体较少，则调低阈值
    score_threshold=.6, #框置信度阈值，小于阈值的框被删除，需要的框较多，则调低阈值，需要的框较少，则调高阈值；
    iou_threshold=.5):  #同类别框的IoU阈值，大于阈值的重叠框被删除，重叠物体较多，则调高阈值，重叠物体较少，则调低阈值
    """

    # 获得特征层的数量
    num_layers = len(yolo_outputs) #yolo的输出层数；num_layers = 3  -> 13、26、52，return Model(inputs, [y1,y2,y3])，y：[m,gird,grid,3,5+1]
    # 特征层1对应的anchor是678
    # 特征层2对应的anchor是345
    # 特征层3对应的anchor是012
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]

    #K.shape(yolo_outputs)-->((N,13,13,3*?)，(N,26,26,3*?)，(N,52,52,3*?))
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32 #input_shape =（416,416）
    boxes = []
    box_scores = []
    # 对每个特征层进行处理
    for l in range(num_layers):
        #将每一个特征层单独作为变量传入
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # 将每个特征层的结果进行堆叠
    boxes = K.concatenate(boxes, axis=0) #将多个层的数据展平，类似于抹低一个维度
    box_scores = K.concatenate(box_scores, axis=0) #将多个层的数据展平，类似于抹低一个维度
    #box_scores-->(n,num_classes)
    mask = box_scores >= score_threshold #判断一个框的score是否大于阈值，大于的话才会被认定是有东西的
    max_boxes_tensor = K.constant(max_boxes, dtype='int32') #K.constant：声明常量tensor
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # 取出所有box_scores >= score_threshold的框，和成绩
        class_boxes = tf.boolean_mask(boxes, mask[:, c]) #mask为True的数据保留下来
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c]) #mask为True的数据保留下来

        # 非极大抑制，去掉box重合程度高的那一些
        # 非极大抑制，去掉box重合程度高的那一些

        """
        原理：
        (1)从最大概率矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个设定的阈值;
        (2)假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记第一个矩形框F，是我们保留下来的。
        (3)从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，重叠度大于一定的阈值，那么就扔掉；并标记E是我们保留下来的第二个矩形框。
        就这样一直重复，找到所有被保留下来的矩形框。输出为一个一位列表，代表哪些框被留下以及排序-->[1,0,5,6,3,...]
        """
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        # 获取非极大抑制后的结果
        # 下列三个分别是
        # 框的位置，得分与种类
        '''
        tf.gather(params,indices,axis=0 )
        从params的axis维根据indices的参数值获取切片
        params=[1,0,3]
        indices=[[1,1],[2,2],[3,3],[4,4]]
        gather_result=[[2,2],[1,1],[4,4]]
        '''
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        '''
        K.ones_like
        给定一个tensor（tensor 参数），
        该操作返回一个具有和给定tensor相同形状（shape）和相同数据类型（dtype），
        但是所有的元素都被设置为1的tensor。也可以为返回的tensor指定一个新的数据类型。
        '''
        classes = K.ones_like(class_box_scores, 'int32') * c #先生成一个与class_box_scores形状也就是剩下的框的数量一致的全为一的向量，通过 * c 来做到与类别序号一致，以赋予剩下的张量的类别
        boxes_.append(class_boxes) #堆叠数据
        scores_.append(class_box_scores) #堆叠数据
        classes_.append(classes) #堆叠数据
    boxes_ = K.concatenate(boxes_, axis=0) #将多个层的数据展平，类似于抹低一个维度
    scores_ = K.concatenate(scores_, axis=0) #将多个层的数据展平，类似于抹低一个维度
    classes_ = K.concatenate(classes_, axis=0) #将多个层的数据展平，类似于抹低一个维度

    return boxes_, scores_, classes_


