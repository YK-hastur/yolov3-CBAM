#----------------------------------------------------#
#   获取测试集的detection-result和images-optional
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
from yolo import YOLO
from PIL import Image, ImageFont, ImageDraw
from keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
from keras.models import load_model
from utils.utils import letterbox_image
from nets.yolo3 import yolo_body,yolo_eval
import colorsys
import numpy as np
import os
class mAP_YOLO(YOLO):
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        self.score = 0.2
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        # 计算anchor数量
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path)
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

        self.input_image_shape = K.placeholder(shape=(2, ))

        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                num_classes, self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image_id, image):
        f = open("input/detection-results/"+image_id+".txt","w")
        # 调整图片使其符合输入要求

        boxed_image = letterbox_image(image, self.model_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # 预测结果
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        for i, c in enumerate(out_classes):
            predicted_class = self.class_names[int(c)]
            score = str(out_scores[i])

            top, left, bottom, right = out_boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))  # 打印在图片中找到了多少框

        font = ImageFont.truetype(font='font/simhei.ttf',  # 设置字体和字号
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300  # 设置目标框线条的宽度

        small_pic = []

        for i, c in list(enumerate(out_classes)):  # 按输出的box类别序列遍历，enumerate()可生成一个对应序号的数组，并与原数组zip()在一起
            predicted_class = self.class_names[c]  # 读取box类别序列序号，并通过序号在文本列表中找到相应的类别文本，相当于将序号翻译为文本
            box = out_boxes[i]  # 读取box框的位置,[top, left, bottom, right]
            score = out_scores[i]  # 读取box框的置信度，[score]

            top, left, bottom, right = box  # 提取box框位置

            # 非必要内容，将框略微扩大
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            """防止检测框溢出"""
            top = max(0, np.floor(top + 0.5).astype('int32'))  # np.floor向下取整，目标框的上、左两个坐标小数点后一位四舍五入，与0值边界相比，取最大值
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))  # 目标框的下、右两个坐标小数点后一位四舍五入，与图片尺寸相比，取最小值
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 画框
            label = '{} {:.2f}'.format(predicted_class, score)  # 定义label文本内容
            draw = ImageDraw.Draw(image)  # 声明画布
            label_size = draw.textsize(label, font)  # 返回label字符串的长宽，以像素为单位
            label = label.encode('utf-8')  # label文本以utf-8编码
            #print(label, (left, top), (right, bottom))  # 打印相关内容

            if top - label_size[1] >= 0:  # 确定label起始点位置，top>=label高时，label能在框上，反之在框内
                text_origin = np.array([left, top - label_size[1]])  # 定义文字起始位置
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):  # 通过thickness控制绘制矩形的数量，达到控制框的线条粗细的目的
                draw.rectangle(  # 绘制一个矩形，用来组成框
                    [left + i, top + i, right - i, bottom - i],  # 矩形的对角线顶点（左上和右下）
                    outline=self.colors[c])  # 边界颜色（框的颜色）
            draw.rectangle(  # 绘制文本框矩形
                [tuple(text_origin), tuple(text_origin + label_size)],  # text_origin为左上点，text_origin + label_size为右下点
                fill=self.colors[c])  # 矩形内部填充
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)  # 绘制文本内容
            del draw  # 关闭画布

        return image

yolo = mAP_YOLO()

image_ids = open('BCCD_Dataset-master/BCCD/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("input"):
    os.makedirs("input")
if not os.path.exists("input/detection-results"):
    os.makedirs("input/detection-results")
if not os.path.exists("input/images-optional"):
    os.makedirs("input/images-optional")
if not os.path.exists("input/images-ditect-result"):
    os.makedirs("input/images-ditect-result")

for image_id in image_ids:
    image_path = "BCCD_Dataset-master/BCCD/JPEGImages/"+image_id+".jpg"
    image = Image.open(image_path)
    # 开启后在之后计算mAP可以可视化
    image.save("input/images-optional/"+image_id+".jpg")
    image_result = yolo.detect_image(image_id,image)
    image_result.save("input/images-ditect-result/" + image_id + ".jpg")
    print(image_id," done!")
    
print("Conversion completed!")
