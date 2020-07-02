import numpy as np
import tensorflow as tf
from keras import backend as K


#---------------------------------------------------#
#   将预测值的每个特征层调成真实值
#---------------------------------------------------#
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    # [1, 1, 1, num_anchors, 2]
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # 获得x，y的网格
    # (13, 13, 1, 2)
    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    # (batch_size,13,13,3,85)
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    '''
    将预测值调成真实值
    box_xy对应框的中心点
    box_wh对应框的宽和高
    bx=sigmoid(tx)+cx
    by=sigmoid(ty)+cy
    bw=pw*e^(tw)
    bh=ph*e^(th)
    b-->box
    t-->true
    c-->box_center
    p-->predict_anchor_box
    '''
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # 在计算loss的时候返回如下参数
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

#---------------------------------------------------#
#   用于计算每个预测框与真实框的iou
#---------------------------------------------------#
def box_iou(b1, b2):
    # 13,13,3,1,4
    # 计算左上角的坐标和右下角的坐标
    b1 = K.expand_dims(b1, -2) #在b1倒数第2个添加1位，即(3,2)->(3,1,2)
    b1_xy = b1[..., 0:2] #读取b1中心的xy
    b1_wh = b1[..., 2:4] #读取b1的宽高
    b1_wh_half = b1_wh/2. #求中心到宽高两边的距离
    b1_mins = b1_xy - b1_wh_half #求左上角顶点坐标
    b1_maxes = b1_xy + b1_wh_half #求右下角顶点坐标

    # 1,n,4
    # 计算左上角和右下角的坐标
    b2 = K.expand_dims(b2, 0) #在b2前端添加一个维度,由(9,2)转为(1,9,2)
    b2_xy = b2[..., :2] #读取b2中心的xy
    b2_wh = b2[..., 2:4] #读取b2的宽高
    b2_wh_half = b2_wh/2. #求中心到宽高两边的距离
    b2_mins = b2_xy - b2_wh_half #求左上角顶点坐标
    b2_maxes = b2_xy + b2_wh_half #求右下角顶点坐标

    # 计算重合面积
    intersect_mins = K.maximum(b1_mins, b2_mins) #对x和y逐位进行比较，选出最大的左上角位置，作为相交区域的左上角点
    intersect_maxes = K.minimum(b1_maxes, b2_maxes) #对x和y逐位进行比较，选出最小的右下角位置，作为相交区域的右下角点
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.) #逐位比对宽高，将负值（无相交）转化为0
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1] #计算相交面积
    b1_area = b1_wh[..., 0] * b1_wh[..., 1] #计算b1的面积
    b2_area = b2_wh[..., 0] * b2_wh[..., 1] #计算b2的面积
    iou = intersect_area / (b1_area + b2_area - intersect_area) #计算iou

    return iou

#---------------------------------------------------#
#   loss值计算
#---------------------------------------------------#
def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    """
    在损失方法yolo_loss中，设置若干参数：
    num_layers：层的数量，是anchors数量的3分之1或2分之1；
    yolo_outputs和y_true：分离args，前3个是yolo_outputs预测值，后3个是y_true真值；
    anchor_mask：anchor box的索引数组，3个1组倒序排序，678对应13x13，345对应26x26，123对应52x52；即[[6, 7, 8], [3, 4, 5], [0, 1, 2]]；
    input_shape：K.shape(yolo_outputs[0])[1:3]，第1个预测矩阵yolo_outputs[0]的结构（shape）的第1~2位，即(?, 13, 13, 18)中的(13, 13)。再x32，就是YOLO网络的输入尺寸，即(416, 416)，因为在网络中，含有5个步长为(2, 2)的卷积操作，降维32=5^2倍；
    grid_shapes：与input_shape类似，K.shape(yolo_outputs[l])[1:3]，以列表的形式，选择3个尺寸的预测图维度，即[(13, 13), (26, 26), (52, 52)]；
    m：第1个预测图的结构的第1位，即K.shape(yolo_outputs[0])[0]，输入模型的图片总量，即批次数；
    mf：m的float类型，即K.cast(m, K.dtype(yolo_outputs[0]))
    loss：损失值为0；
    yolo_outputs: [(?, 13, 13, 18), (?, 26, 26, 18), (?, 52, 52, 18)]
    y_true: [(?, 13, 13, 18), (?, 26, 26, 18), (?, 52, 52, 18)]
    args: [model_body,y_true]-->[(?, 13, 13, 18), (?, 26, 26, 18), (?, 52, 52, 18), (?, 13, 13, 18), (?, 26, 26, 18), (?, 52, 52, 18)]
    """

    num_layers = len(anchors)//3 #求anchor层数

    # 将预测结果和实际ground truth分开，args是[*model_body.output, *y_true]
    # y_true是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    # yolo_outputs是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    y_true = args[num_layers:] #真值
    yolo_outputs = args[:num_layers] #模型输出值

    # 先验框
    # 678为116,90,  156,198,  373,326
    # 345为30,61,  62,45,  59,119
    # 012为10,13,  16,30,  33,23,  
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] #确定anchor的分布

    # 得到input_shpae为416,416 
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0])) #K.cast：转换数据格式，input_shape-->(416,416)

    # 得到网格的shape为13,13;26,26;52,52
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)] #grid_shapes-->[(13,13),(26,26),(52,52)]
    loss = 0 #loss值

    # 取出每一张图片
    # m的值就是batch_size
    m = K.shape(yolo_outputs[0])[0] #batch值
    mf = K.cast(m, K.dtype(yolo_outputs[0])) #转换为float格式

    # y_true是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    # yolo_outputs是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    for l in range(num_layers):
        # 以第一个特征层(m,13,13,3,85)为例子
        object_mask = y_true[l][..., 4:5] #object_mask是y_true的第4位，即是否含有物体，含有是1，不含是0
        true_class_probs = y_true[l][..., 5:] # 取出其对应的种类(m,13,13,3,80)

        # 将yolo_outputs的特征层输出进行处理
        # grid为网格结构(13,13,1,2)，raw_pred为尚未处理的预测结果(m,13,13,3,85)
        # 还有解码后的xy，wh，(m,13,13,3,2)
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]],
                                                     num_classes,
                                                     input_shape,
                                                     calc_loss=True)
        
        # 这个是解码后的预测的box的位置
        # (m,13,13,3,4)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # 找到负样本群组，第一步是创建一个数组，[]
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True) #建立一个长度可变的空数组
        object_mask_bool = K.cast(object_mask, 'bool') #转换数据格式，(m,13,13,3,1)
        
        # 对每一张图片计算ignore_mask
        def loop_body(b, ignore_mask):
            # 取出第b副图内，真实存在的所有的box的参数
            # n,4
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0]) #object_mask_bool为True的y_true数据保留下来，挑选出置信度大于0的框的相应的坐标，truebox形式为中心坐标xy与hw
            # 计算预测结果与真实情况的iou
            # pred_box为(13,13,3,4)
            # 计算的结果是每个pred_box和其它所有真实框的iou
            # iou-->(13,13,3,n)
            iou = box_iou(pred_box[b], true_box)

            # best_iou-->(13,13,3,1)
            best_iou = K.max(iou, axis=-1) #将iou降序排序

            # 判断预测框的iou小于ignore_thresh则认为该预测框没有与之对应的真实框
            # 则被认为是这幅图的负样本
            ignore_mask = ignore_mask.write(index=b, value=K.cast(best_iou<ignore_thresh, K.dtype(true_box))) #在指定的位置（index）写入value（tensor）
            return b+1, ignore_mask

        # 遍历所有的图片
        '''
        循环处理某个变量，中间处理的结果用来进行下一次处理，最后输出经过数次加工的变量
        lambda b,*args: b<m：是条件函数。lambda 是匿名函数关键字。b,*args是形参，b<m是返回的结果
        loop_body：是循环目标函数。
        [0, ignore_mask]：是函数的起始实参。
        条件函数和循环函数的参数相同。
        '''
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, #循环的判断条件，输入与下面的函数主体一致
                                                       loop_body, #循环的函数主体
                                                       [0, ignore_mask]) #循环的初始输入

        # 将每幅图的内容压缩，进行处理
        ignore_mask = ignore_mask.stack() #将一个列表中维度数目为R的张量堆积起来形成维度为R+1的新张量，相当于对TensorArray进行解码操作，让TensorArray中的变量全部聚合到一个矩阵里
        ignore_mask = K.expand_dims(ignore_mask, -1) #增加维度，ignore_mask-->(m,13,13,3,1,1)

        # 将真实框进行编码，使其格式与预测的相同，后面用于计算loss
        raw_true_xy = y_true[l][..., 0:2]*grid_shapes[l][::-1] - grid #将真值转化为相对于网格中心点的xy，偏移数据，值的范围是0~1，是逆yolo_head的操作
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1]) #将真实的wh转换为相对于anchors的wh的比例，是逆yolo_head的操作

        '''
        object_mask如果真实存在目标则保存其wh值
        K.switch(条件函数，true返回值，else返回值)其中1,2要等shape
        K.zeros_like，返回与输入矩阵形状一致的全零矩阵
        对raw_true_wh进行清洗，将不含有box的位置都置为零，避免出现由上一步log(0)=-inf产生的这种异常值
        '''
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))

        '''
        提升针对小物体的小技巧：针对 YOLOv3来说，regression损失会乘一个（2-w*h）的比例系数，
        以提升小物体在loss中的影响力。
        w 和 h 分别是ground truth 的宽和高。如果不减去 w*h，AP 会有一个明显下降。
        如果继续往上加，如 (2-w*h)*1.5，总体的 AP 还会涨一个点左右（包括验证集和测试集），
        大概是因为 COCO 中小物体实在太多的原因。
        '''
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        '''
        xy_loss：中心点的损失值。object_mask是y_true的第4位，即是否含有物体，含有是1，不含是0。
                 box_loss_scale的值，与物体框的大小有关,提升小物体在loss中的影响力。2减去相对面积，值得范围是(1~2)。
                 binary_crossentropy是二值交叉熵。当模型最后一层没有经过激活函数时 from_logits 设置为 True，否则为 False
        wh_loss：宽高的损失值。除此之外，额外乘以系数0.5，平方K.square()。
        '''
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])

        '''
        confidence_loss：框置信度的损失值。两部分组成，第1部分是存在物体的损失值，第2部分是不存在物体的损失值，其中乘以忽略掩码ignore_mask，忽略预测框中IoU大于阈值的框。
        class_loss：类别损失值。
        将各部分损失值的和，除以均值，累加，作为最终的图片损失值。
        如果该位置本来有框，那么计算1与置信度的交叉熵
        如果该位置本来没有框，而且满足预测框与真实框的iou小于忽略阈值(best_iou<ignore_thresh)，则被认定为负样本
        best_iou<ignore_thresh用于限制负样本数量
        '''
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)


        xy_loss = K.sum(xy_loss) / mf #将该loss除以输入的图片数量，求得单位图像第l层的该损失值
        wh_loss = K.sum(wh_loss) / mf #同上
        confidence_loss = K.sum(confidence_loss) / mf #同上
        class_loss = K.sum(class_loss) / mf #同上
        loss += xy_loss + wh_loss + confidence_loss + class_loss #对损失值进行累加
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ') #打印loss
    return loss #返回累加了所有anchor尺度层后的loss值