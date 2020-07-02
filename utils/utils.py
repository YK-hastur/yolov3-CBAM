"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    #本函数的目的是将图片缩放并在缩放后的空白地区填充灰色像元
    iw, ih = image.size #原图像尺寸
    w, h = size #处理后尺寸，（416,416）
    scale = min(w/iw, h/ih) #求缩放比例，只缩不扩
    nw = int(iw*scale) #将图像长宽同步缩放
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC) #使用BICUBIC方法将原图像重采样至缩放后尺寸
    new_image = Image.new('RGB', size, (128,128,128)) #建立一个处理后尺寸的全灰图像
    new_image.paste(image, ((w-nw)//2, (h-nh)//2)) #将缩放后图像放置近全灰图像中，并保证在处理后图像长宽的中心
    return new_image

def rand(a=0.0, b=1.0):
    return np.random.rand()*(b-a) + a

def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):

    """
    r实时数据增强的随机预处理
    annotation_lines：标注数据的行，每行数据包含图片路径，和框的位置信息，种类：（'G:\日常文件\4语义分割\yolo3-keras-master\VOCdevkit\VOC2007\JPEGImages\000017.jpg 185,62,279,199,14 90,78,403,336,12'）
    return:imagedata是经过resize并填充的样本图片，resize成（416,416），并填充灰度
    boxdata是每张image中做的标记label，shpe，对应着truebox，批次数8，最大框数20，每个框5个值，4个边界点和1个类别序号，如(16, 20, 5)为（batchsize，maxbox，5），每张图片最多的有maxbox个类，5为左上右下的坐标
    """

    line = annotation_line.split() #分割line文本：('G:\日常文件\4语义分割\yolo3-keras-master\VOCdevkit\VOC2007\JPEGImages\000017.jpg', '185,62,279,199,14', '90,78,403,336,12')
    image = Image.open(line[0]) #打开图片 'G:\日常文件\4语义分割\yolo3-keras-master\VOCdevkit\VOC2007\JPEGImages\000017.jpg'
    iw, ih = image.size #读取原图片尺寸
    h, w = input_shape #读取输入图片尺寸

    '''
    for box in line[1:]跳过图片路径，从第一个box开始逐个box进行分隔
    box.split(',')以','为分隔符，将剩余的字符串分为('185','62','279','199','14'),('90','78','403','336','12')
    map函数进行取整，然后将list列表转化为数组
    '''

    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]]) #读取box数据并按','分割

    if not random: #将图片等比例转换为416x416的图片，其余用灰色填充，并做归一化
        # resize image
        scale = min(w/iw, h/ih) #求缩放比例，只缩不扩
        nw = int(iw*scale) #将图像长宽同步缩放
        nh = int(ih*scale)
        dx = (w-nw)//2 #x方向偏移量
        dy = (h-nh)//2 #y方向偏移量
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC) #使用BICUBIC方法将原图像重采样至缩放后尺寸
            new_image = Image.new('RGB', (w,h), (128,128,128)) #建立一个处理后尺寸的全灰图像
            new_image.paste(image, (dx, dy)) #将缩放后图像放置近全灰图像中，并保证在处理后图像长宽的中心
            image_data = np.array(new_image)/255. #归一化

        # correct boxes
        box_data = np.zeros((max_boxes,5)) #声明一个全为0的数组
        if len(box)>0:
            np.random.shuffle(box) #打混box排列顺序，为下一步有可能的随机取box做准备
            if len(box)>max_boxes: box = box[:max_boxes] #取出最多max_boxes个box
            box[:, [0,2]] = box[:, [0,2]]*scale + dx #box坐标与图片同步进行缩放和偏移
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box #将处理好的box传入box_data

        #image_data-->(416,416,3),box_data-->(20,5)
        return image_data, box_data
    '''
    resize image
    通过jitter参数，随机计算new_ar和scale，生成新的nh和nw，
    将原始图像随机转换为nw和nh尺寸的图像，即非等比例变换图像,属于数据增强
    '''
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter) #新的随机宽高比
    scale = rand(0.25, 2) #新的缩放比
    if new_ar < 1:
        nh = int(scale*h) #缩放h
        nw = int(nh*new_ar) #通过新的宽高比计算w
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC) #使用BICUBIC方法将原图像重采样至缩放后尺寸

    # place image
    dx = int(rand(0, w-nw)) #保证图像不移出边界的条件下，随机生成x偏移量
    dy = int(rand(0, h-nh)) #保证图像不移出边界的条件下，随机生成y偏移量
    new_image = Image.new('RGB', (w,h), (128,128,128)) #建立一个处理后尺寸的全灰图像
    new_image.paste(image, (dx, dy)) #将缩放后图像放置近全灰图像中，并保证在处理后图像长宽的中心
    image = new_image

    # flip image or not
    flip = rand()<.5 #有50%的几率翻转图片
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    '''
    distort image
    在HSV色域中，改变图片的颜色范围，hue值相加，sat和vat相乘，色调（H），饱和度（S），明度（V）
    先由RGB转为HSV，再由HSV转为RGB，添加若干错误判断，避免范围过大
    '''
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.) #将image由RGB转为HSV，并做归一化
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1 #在多维数组中，使x[..., 0]中 >1 的值减1，以保证x[..., 0]中所有数值均在0-1之间
    x[..., 0][x[..., 0]<0] += 1 #在多维数组中，使x[..., 0]中 <1 的值加1，以保证x[..., 0]中所有数值均在0-1之间
    x[..., 1] *= sat #饱和度（S）乘上偏移量
    x[..., 2] *= val #明度（V）乘上偏移量
    x[x>1] = 1 #再次判断，使x中所有数值均在0-1之间
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1，由HSV转为RGB

    # correct boxes
    box_data = np.zeros((max_boxes,5)) #声明一个全为0的数组
    if len(box)>0:
        np.random.shuffle(box) #打混box排列顺序，为下一步有可能的随机取box做准备
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx #box坐标与图片同步进行缩放和偏移
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]] #若图像翻转了，将box坐标也进行翻转
        box[:, 0:2][box[:, 0:2]<0] = 0 #判断box左上角点是否低于0，也就是在图像外，若是则将点移到（0,0）
        box[:, 2][box[:, 2]>w] = w #判断box右边界是否大于w，也就是在图像外，若是则将边移到w
        box[:, 3][box[:, 3]>h] = h #判断box下边界是否大于h，也就是在图像外，若是则将边移到h
        box_w = box[:, 2] - box[:, 0] #计算box宽
        box_h = box[:, 3] - box[:, 1] #计算box高
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box，np.logical_and为‘与’操作，即box的宽高均大于1才会返回True，舍弃不合格的box
        if len(box)>max_boxes: box = box[:max_boxes] #取出最多max_boxes个box
        box_data[:len(box)] = box #将处理好的box传入box_data

    return image_data, box_data
    
def print_answer(argmax):
    with open("./model_data/index_word.txt","r",encoding='utf-8') as f:
        synset = [l.split(";")[1][:-1] for l in f.readlines()]
        
    return synset[argmax]