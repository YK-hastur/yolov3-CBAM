import os
import numpy as np
import copy
import colorsys
from timeit import default_timer as timer
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from nets.yolo3 import yolo_body,yolo_eval,tiny_yolo_body
from utils.utils import letterbox_image

#---------------------------------------------------#
#   本文件用于载入已训练好的模型，对预测结果进行解码并绘制预测后结果
#---------------------------------------------------#

class YOLO(object):
    _defaults = {
        "model_path": 'logs/ep100-loss82.796-val_loss87.608.h5',
        "anchors_path": 'model_data/yolo_anchors_bccd.txt',
        "classes_path": 'yolo_classes.txt',
        "score" : 0.5,
        "iou" : 0.3,
        "model_image_size" : (480, 480)
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化yolo
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) #18行，用于定义文件路径、图像尺寸等初始值，并实例化成self变量
        self.class_names = self._get_class() #47行，函数，获取当前数据集的类别
        self.anchors = self._get_anchors() #52行，函数，获取anchor框的尺寸
        self.sess = K.get_session() #创建一个新的全局会话
        self.boxes, self.scores, self.classes = self.generate() #

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path) #转化为绝对路径
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names] #strip()，删除首尾除数字字母外的字符
        return class_names

    #---------------------------------------------------#
    #   获得所有的先验框
    #---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        # 计算anchor数量
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2, )) #声明一个定义输入图片尺寸的张量

        """      
        yolo_outputs        #模型输出，格式如下[（?，13,13,255）（?，26,26,255）（?,52,52,255）] ?:bitch size; 13、26、52:多尺度预测； 255：预测值（3*（80+5））
        anchors,            #[(10,13)，(16,30), (33,23), (30,61), (62,45), (59,119), (116,90), (156,198),(373,326)]
        num_classes,        #类别个数，coco集80类
        image_shape,        #placeholder类型的TF参数，默认(416, 416)；
        max_boxes=20,       #每张图每类最多检测到20个框同类别框的IoU阈值，大于阈值的重叠框被删除，重叠物体较多，则调高阈值，重叠物体较少，则调低阈值
        score_threshold=.6, #框置信度阈值，小于阈值的框被删除，需要的框较多，则调低阈值，需要的框较少，则调高阈值；
        iou_threshold=.5):  #同类别框的IoU阈值，大于阈值的重叠框被删除，重叠物体较多，则调高阈值，重叠物体较少，则调低阈值
        """
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                num_classes, self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image): #image为PIL打开图片后的对象，而非矩阵

        """
        开始计时->
        1.调用letterbox_image函数，即：
        先生成一个用“绝对灰”R128-G128-B128填充的416×416新图片，
        然后用按比例缩放（采样方式：BICUBIC）后的输入图片粘贴，
        粘贴不到的部分保留为灰色。
        2.model_image_size定义的宽和高必须是32的倍数；
        若没有定义model_image_size，将输入的尺寸调整为32的倍数，并调用letterbox_image函数进行缩放。
        3.将缩放后的图片数值除以255，做归一化。
        4.将（416,416,3）数组调整为（1,416,416,3）数组，满足网络输入的张量格式：image_data。

        1.运行self.sess.run（）输入参数：输入图片416×416，学习模式0测试/1训练。
 66     self.yolo_model.input: image_data，self.input_image_shape: [image.size[1], image.size[0]]，
 67     K.learning_phase(): 0。
        2.self.generate（），读取：model路径、anchor box、coco类别、加载模型yolo.h5.，对于80中coco目标，确定每一种目标框的绘制颜色，即：
        将（x/80,1.0,1.0）的颜色转换为RGB格式，并随机调整颜色一遍肉眼识别，其中：一个1.0表示饱和度，一个1.0表示亮度。
        3.若GPU>2调用multi_gpu_model()

        1.yolo_eval(self.yolo_model.output),max_boxes=20,每张图没类最多检测20个框。
 70     2.将anchor_box分为3组，分别分配给三个尺度，yolo_model输出的feature map
 71     3.特征图越小，感受野越大，对大目标越敏感，选大的anchor box->
 72     分别对三个feature map运行out_boxes, out_scores, out_classes，返回boxes、scores、classes。
        """

        start = timer()

        # 调整图片使其符合输入要求
        new_image_size = (self.model_image_size[0],self.model_image_size[1]) #（416,416）
        boxed_image = letterbox_image(image, new_image_size) #本函数的目的是将图片缩放并在缩放后的空白地区填充灰色像元
        image_data = np.array(boxed_image, dtype='float32') #转换数据类型
        image_data /= 255. #归一化
        image_data = np.expand_dims(image_data, 0)  #添加batch维，(416,416)-->(1,416,416)

        # 预测结果
        out_boxes, out_scores, out_classes = self.sess.run( #在generate()内已经载入计算图结构，使用sess.run()运行计算图，返回的是generate()的结果
            [self.boxes, self.scores, self.classes], #使返回的值具有与fetches参数相同的形状，其中叶子被TensorFlow返回的相应值替换
            feed_dict={ #替换计算图中的特定层的值
                self.yolo_model.input: image_data, #将input替换为image_data
                self.input_image_shape: [image.size[1], image.size[0]], #将generate()中声明的input_image_shape的值替换为(416,416)
                K.learning_phase(): 0 #学习模式 0：预测模型。 1：训练模型
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img')) #打印在图片中找到了多少框

        font = ImageFont.truetype(font='font/simhei.ttf', # 设置字体和字号
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300 # 设置目标框线条的宽度

        small_pic=[]

        for i, c in list(enumerate(out_classes)): #按输出的box类别序列遍历，enumerate()可生成一个对应序号的数组，并与原数组zip()在一起
            predicted_class = self.class_names[c] #读取box类别序列序号，并通过序号在文本列表中找到相应的类别文本，相当于将序号翻译为文本
            box = out_boxes[i] #读取box框的位置,[top, left, bottom, right]
            score = out_scores[i] #读取box框的置信度，[score]

            top, left, bottom, right = box #提取box框位置

            #非必要内容，将框略微扩大
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            """防止检测框溢出"""
            top = max(0, np.floor(top + 0.5).astype('int32')) #np.floor向下取整，目标框的上、左两个坐标小数点后一位四舍五入，与0值边界相比，取最大值
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32')) #目标框的下、右两个坐标小数点后一位四舍五入，与图片尺寸相比，取最小值
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 画框
            label = '{} {:.2f}'.format(predicted_class, score) #定义label文本内容
            draw = ImageDraw.Draw(image) #声明画布
            label_size = draw.textsize(label, font) #返回label字符串的长宽，以像素为单位
            label = label.encode('utf-8') #label文本以utf-8编码
            print(label, (left, top), (right, bottom)) #打印相关内容
            
            if top - label_size[1] >= 0: #确定label起始点位置，top>=label高时，label能在框上，反之在框内
                text_origin = np.array([left, top - label_size[1]]) #定义文字起始位置
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness): #通过thickness控制绘制矩形的数量，达到控制框的线条粗细的目的
                draw.rectangle( #绘制一个矩形，用来组成框
                    [left + i, top + i, right - i, bottom - i], #矩形的对角线顶点（左上和右下）
                    outline=self.colors[c]) #边界颜色（框的颜色）
            draw.rectangle( #绘制文本框矩形
                [tuple(text_origin), tuple(text_origin + label_size)], #text_origin为左上点，text_origin + label_size为右下点
                fill=self.colors[c]) #矩形内部填充
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font) #绘制文本内容
            del draw #关闭画布

        end = timer() #结束计时
        print(end - start) #打印运行时间
        return image

    def close_session(self):
        self.sess.close() #删除图结构
