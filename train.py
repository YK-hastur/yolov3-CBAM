import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from nets.yolo3 import yolo_body,tiny_yolo_body
from nets.loss import yolo_loss
from keras.backend.tensorflow_backend import set_session
from utils.utils import get_random_data


#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
def get_classes(classes_path): #获取类别文本
    """loads the classes"""
    with open(classes_path) as f:
        class_names = f.readlines() #读取文件并按行载入
    class_names = [c.strip() for c in class_names] #.strip()为清理文本首尾字符
    return class_names

def get_anchors(anchors_path): #获取anchor
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline() #读取文件并按行载入
    anchors = [float(x) for x in anchors.split(',')] #以','为标志分割文本
    return np.array(anchors).reshape(-1, 2)

#---------------------------------------------------#
#   训练数据生成器
#---------------------------------------------------#
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes): #fit_generator，用于训练时向模型输入数据
    '''data generator for fit_generator'''
    n = len(annotation_lines) #读取样本数
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size): #取batch_size次数据
            if i==0: #i==0时意味着所有样本已经都被使用过了，需要重新开始新一轮读取
                np.random.shuffle(annotation_lines) #将样本顺序打混
            image, box = get_random_data(annotation_lines[i], input_shape, random=True) #读取数据并进行数据增强,image-->(416,416,3),box-->(20,5)
            image_data.append(image) #向image_data内添加处理过的image
            box_data.append(box) #向box_data内添加处理过的box
            i = (i+1) % n #判定的核心程序，当i+1==n时，即所有样本都读取过了，(i+1) % n则=0
        image_data = np.array(image_data) #将image_data转化为numpy格式
        box_data = np.array(box_data) #将box_data转化为numpy格式
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes) #通过box_data和anchors计算y_true，#将真实坐标转化为yolo需要输入的坐标
        yield [image_data, *y_true], np.zeros(batch_size) #[image_data, *y_true]为真正的输入项，np.zeros(batch_size)为占位置的向量，不参与计算


#---------------------------------------------------#
#   读入xml文件，并输出y_true
#---------------------------------------------------#
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """
    true_boxes：box，批次数8，最大框数20，每个框5个值，4个边界点和1个类别序号，如(8, 20, 5)
    input_shape：图片尺寸，如(416, 416)
    anchors：anchor列表,(9,2)
    num_classes：类别的数量,int

    """
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes' #检查有无异常数据 即txt提供的box id 是否存在大于 num_class的情况
    # 一共有三个特征层数
    num_layers = len(anchors)//3 #输出的层数

    '''
    先验框
    [6,7,8]为116,90  156,198  373,326 -->(8,13,13,3,25)
    [3,4,5]为 30,61   62, 45   59,119 -->(8,26,26,3,25)
    [0,1,2]为 10,13   16, 30   33, 23 -->(8,52,52,3,25)
    '''
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] #判断是否为tiny模型，tiny模型只有两个尺度层

    true_boxes = np.array(true_boxes, dtype='float32') #转换格式，方便计算
    input_shape = np.array(input_shape, dtype='int32') # 416,416
    # 读出xy轴，读出长宽
    # 中心点(m,n,2)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2 #通过左上和右下的顶点，计算box的中心坐标
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2] #通过左上和右下的顶点，计算box的宽高
    # 计算比例
    true_boxes[..., 0:2] = boxes_xy/input_shape[:] #计算坐标在整图像中的百分比坐标
    true_boxes[..., 2:4] = boxes_wh/input_shape[:] #计算宽高在整图像中的百分比宽高

    # m张图
    m = true_boxes.shape[0] #获取batch数
    # 得到网格的shape为13,13;26,26;52,52
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)] #获取网格数量，标准模型获取三个数量，tiny模型获取前两个数量
    # y_true的标准模型格式为[(m,13,13,3,25),(m,26,26,3,25),(m,52,52,3,25)]，tiny模型格式为[(m,13,13,3,25),(m,26,26,3,25)]的全零矩阵
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]
    # [1,9,2]
    anchors = np.expand_dims(anchors, 0) #在原先axis出添加一个维度,由(9,2)转为(1,9,2)
    anchor_maxes = anchors / 2. #网格中心为原点（即网格中心坐标为（0,0））,　计算出anchor框内的右下角坐标
    anchor_mins = -anchor_maxes #计算出左上角坐标

    '''
    将boxes_wh中宽w大于0的位，设为True，即含有box，结构是(16,20)
    1 True,True,True, False,False,False,False,False,False...
    2 True,True, False,False,False,False,False,False,False...
    3 True,True,True,True,True,True, False,False,False...
    ...
    '''
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # 对每一张图进行处理
        wh = boxes_wh[b, valid_mask[b]] #只选择存在标注框的wh,例如：wh_1的shape是(3,2)
        if len(wh)==0: continue #该图中所有的box都不合格
        wh = np.expand_dims(wh, -2) #在wh倒数第2个添加1位，即(3,2)->(3,1,2)
        box_maxes = wh / 2. #box中心为原点（即box中心坐标为（0,0））,　计算出box内的右下角坐标，与anchor计算一致
        box_mins = -box_maxes #计算出box左上角坐标

        '''
        通过计算真实值和anchor的IOU，筛选出和真实框最契合的先验框
        box_mins的shape是(3,1,2)，anchor_mins的shape是(1,9,2)，intersect_mins的shape是(3,9,2)，即两两组合的值
        intersect_area的shape是(3,9)；
        box_area的shape是(3,1)；
        anchor_area的shape是(1,9)
        iou的shape是(3,9)
        best_anchor的shape是(3,1)
        '''
        intersect_mins = np.maximum(box_mins, anchor_mins) #对x和y逐位进行比较，选出最大的左上角位置，作为相交区域的左上角点
        intersect_maxes = np.minimum(box_maxes, anchor_maxes) #对x和y逐位进行比较，选出最小的右下角位置，作为相交区域的右下角点
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.) #逐位比对宽高，将负值（无相交）转化为0
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1] #计算相交面积
        box_area = wh[..., 0] * wh[..., 1] #计算box的面积
        anchor_area = anchors[..., 0] * anchors[..., 1] #计算anchor的面积
        iou = intersect_area / (box_area + anchor_area - intersect_area) #计算iou
        best_anchor = np.argmax(iou, axis=-1) #寻找每个box最好的iou，并返回该iou的索引

        '''
        t是box的序号；n是最优anchor的序号；
        l是层号；如果最优anchor在层l中，则设置其中的值，否则默认为0；anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
        true_boxes：box，批次数8，最大框数20，每个框5个值，x、y、w、h和1个类别序号，如(8, 20, 5)
        true_boxes[b, t, 0]，其中b是批次序号、t是box序号，第0位是x，第1位是y；
        grid_shapes是3个检测图的尺寸，将归一化的值，与框长宽相乘，恢复为具体值；
        k是在anchor box中的序号；
        c是类别，true_boxes的第4位；
        将xy和wh放入y_true中，将y_true的第4位框的置信度设为1，第5~n位的类别设为1；
        '''
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):# 遍历anchor的3个尺寸（13,26,52）,因为此时box已经和一个anchor box匹配上，看这个anchor box属于哪一层，小，中，大，然后将其box分配到那一层
                if n in anchor_mask[l]:
                    #np.floor用于向下取整，np.floor的返回值是不大于输入参数的最大整数。
                    #i，j是所在网格的位置
                    #第b个图像，第t个box的x乘以第l个grid shap的x
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][0]).astype('int32') #true_boxes[b,t,0]为0-1的数值，grid_shapes[l][1]为52、26或13，用于计算box位于第几个网格内
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][1]).astype('int32')
                    k = anchor_mask[l].index(n) #找到n在anchor层中的索引位置,index()函数用于从列表中找出某个值第一个匹配项的索引位置，以找到真实框在特征层l中第b副图像对应的位置
                    c = true_boxes[b,t, 4].astype('int32') #得到box所属的类别
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4] #第b个图像，在anchor层的第j行i列的第k尺寸的anchor中，储存box相对于整幅图像的x、y、w、h
                    y_true[l][b, j, i, k, 4] = 1 #将物体置信度设为1
                    y_true[l][b, j, i, k, 5+c] = 1 #将所属类别置信度设为1

    return y_true #对于输出的y_true而言，只有每个图里每个框最对应的位置有数据，其它的地方都为0


#config = tf.ConfigProto() #配置tf.Session的运算方式，比如gpu运算或者cpu运算
#config.gpu_options.allocator_type = 'BFC' #用于设置GPU分配的策略。”BFC”指定采用最佳适配合并算法
#config.gpu_options.per_process_gpu_memory_fraction = 0.7 #数值在0到1之间，表示预分配多少比例的可用GPU显存给每个进程。比如1表示预分配所有的可用的GPU显存，0.5则表示分配50%的可用的GPU显存。
#config.gpu_options.allow_growth = True #是否采用增长的方式分配显存。如果这个值为True，那么分配器不会预分配整个指定的GPU显存空间，而是开始分配一小部分显存，然后随着需要而增加。
#set_session(tf.Session(config=config)) #载入设置

if __name__ == "__main__":
    # 标签的位置
    annotation_path = 'BCCD_train.txt' #由voc_annotation.py生成，其信息为：'D:\\VOC\\train\\VOCdevkit\\VOC2007\\JPEGImages\\000017.jpg 185,62,279,199,14 90,78,403,336,12'
    # 获取classes和anchor的位置
    val_path = 'BCCD_val.txt'
    classes_path = 'yolo_classes.txt'
    anchors_path = 'model_data/yolo_anchors_bccd.txt'
    # 预训练模型的位置
    weights_path = 'model_data/yolo_weights.h5'
    # 获得classes和anchor
    class_names = get_classes(classes_path) #class_names为列表
    anchors = get_anchors(anchors_path) ##anchors为数组

    load_pretrained = False #是否载入训练好的权重

    num_classes = len(class_names) # 一共有多少类
    num_anchors = len(anchors) # 一共有多少anchor
    # 训练日志保存的位置
    model_weight_dir = 'logs/'
    # 输入的shape大小
    input_shape = (640,640)
    is_tiny_version = len(anchors) == 6  #判断是否为tiny模型

    # 清除session
    K.clear_session()

    # 输入的图像为
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape #长宽赋值

    # 创建yolo模型
    if is_tiny_version:
        print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
        model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes) #调用yolo3.py的tiny_yolo_body生成yolo计算图
        # y_true为[(13, 13, 2, num_classes+5), (26, 26, 2, num_classes+5)]两种大小，
        # 13,13,3,85
        # 26,26,3,85
        # 13,26是grid,3是每个尺度的anchor个数,num_classes+5是框的4个坐标信息,objectness score一位,类别1位和类别数
        y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l], num_anchors // 2, num_classes + 5)) for l in range(2)]
    else:
        print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
        model_body = yolo_body(image_input, num_anchors//3, num_classes) #调用yolo3.py的yolo_body生成yolo计算图
        # y_true为[(13, 13, 3, num_classes+5), (26, 26, 3, num_classes+5), (52, 52, 3, num_classes+5)]三种大小，
        # 13,13,3,85
        # 26,26,3,85
        # 52,52,3,85
        # 13,26,52是grid,3是每个尺度的anchor个数,num_classes+5是框的4个坐标信息,objectness score一位,类别1位和类别数
        y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], num_anchors // 3, num_classes + 5)) for l in range(3)]
    
    # 载入预训练权重
    if load_pretrained:  # 加载预训练值
        print('Load weights {}.'.format(weights_path))
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True) #读取模型权重，skip_mismatch=True-->没有权重文件时跳过载入
        freeze_layers = 184
        for i in range(freeze_layers): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))

    # 输入为*model_body.input, *y_true
    #model_body: [(?, 13, 13, 18), (?, 26, 26, 18), (?, 52, 52, 18)]
    #y_true: [(?, 13, 13, 18), (?, 26, 26, 18), (?, 52, 52, 18)]
    loss_input = [*model_body.output, *y_true] #输入到loss函数的数据，*model_body.output-->[y1,y2,y3]即三个尺度的预测结果,每个y都是[m,grid,grid,num_anchors,(num_classes+5)]
    # 输出为model_loss
    # Lambda为keras.layers函数，用于封装函数
    model_loss = Lambda(yolo_loss, #function：封装的函数，用于实现功能
                        output_shape=(1,), #预期的函数输出尺寸
                        name='yolo_loss', #该层的名称
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5} #需要传递给函数的关键字参数，详见loss.py的yolo_loss函数
                        )(loss_input) #输入到封装函数内的数据

    model = Model(inputs=[model_body.input, *y_true], outputs=model_loss) #实例化model
    print(model.summary())
    # 训练参数设置
    logging = TensorBoard(log_dir=model_weight_dir) #该回调函数将日志信息写入TensorBorad，可以动态的观察训练和测试指标的图像以及不同层的激活值直方图。
    checkpoint = ModelCheckpoint(model_weight_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', #该回调函数将在每个epoch后保存模型到filepath
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=2)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1) #当评价指标不在提升时，减少学习率
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1) #当监测值不再改善时，该回调函数将中止训练

    #val_split = 0.1 # 0.1用于验证，0.9用于训练
    with open(annotation_path) as f:
        lines = f.readlines()
    with open(val_path) as f:
        lines_val = f.readlines()

    num_val = int(len(lines_val))
    num_train = len(lines)
    np.random.seed(10101) #赋予随机种子
    np.random.shuffle(lines) #打乱排序
    np.random.seed(None)
    num_val = int(len(lines_val)) #生成验证文件列表
    num_train = len(lines) #生成训练文件列表
    
    # 调整非主干模型first
    if True:
        # loss要求有两个输入端，但模型的输出就是loss,即y_pred就是loss,因而无视y_true.
        # 训练的时候,随便添加一个符合形状的y_true数组即可，在data_generator中，y_true的输入就为np.zeros(batch_size)，是一个无意义向量
        # y_true的真正有意义的输入是在[image_data, *y_true]中的*y_true中，是作为训练用数据一起传入的
        model.compile(optimizer=Adam(lr=1e-3), loss={
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 3 #batch尺寸
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator(lines, batch_size, input_shape, anchors, num_classes), #训练数据生成器
                steps_per_epoch=max(1, num_train//batch_size), #每个epoch训练几个batch
                validation_data=data_generator(lines_val, batch_size, input_shape, anchors, num_classes), #验证数据生成器
                validation_steps=max(1, num_val//batch_size),
                epochs=100, #训练多少epoch
                initial_epoch=0,
                callbacks=[logging, checkpoint]) #回调函数，在每个epoch末尾激活
        model.save_weights(model_weight_dir + 'trained_weights_stage_1.h5') #保存权重文件

    for i in range(freeze_layers): model_body.layers[i].trainable = True

    # 解冻后训练
    if True:
        model.compile(optimizer=Adam(lr=1e-4), loss={
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 4
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=100,
                initial_epoch=50,
                callbacks=[logging, checkpoint])
        model.save_weights(model_weight_dir + 'last1.h5')
