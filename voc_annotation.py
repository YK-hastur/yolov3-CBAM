import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert_annotation(year, image_id, list_file):
    in_file = open('G:\\日常文件\\4语义分割\\yolo3-keras-master\\VOCdevkit\\VOC%s\\Annotations\\%s.xml'%(year, image_id)) #打开xml
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd() #返回当前工作目录的绝对路径

for year, image_set in sets:
    image_ids = open('G:\\日常文件\\4语义分割\\yolo3-keras-master\\VOCdevkit\\VOC%s\\ImageSets\\Main\\%s.txt'%(year, image_set)).read().strip().split() #.read()：打开txt，.strip()：删除首尾空格，.split()：将数据分行
    list_file = open('%s_%s.txt'%(year, image_set), 'w') #在工作目录建立txt文档
    for image_id in image_ids:
        list_file.write('%s\\VOCdevkit\\VOC%s\\JPEGImages\\%s.jpg'%(wd, year, image_id)) #写入图片的绝对路径
        convert_annotation(year, image_id, list_file) #写入box参数
        list_file.write('\n') #换行
    list_file.close() #写入完毕
    print('%05s file with %04d lines done.'%(image_set,len(image_ids)))

print('work done.')
