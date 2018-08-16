# coding=utf-8



#from pylab import *
import os 
import cv2
import random


'''


xml_path = 'E:\\Image\\shandong\\primary_sample_xml\\primary_sample_xml'
img_path = 'E:\\Image\\shandong'
img_save_path = 'E:\\Image\\shandong\\primary_sample_xml'


list = os.listdir(xml_path)


for filename in list:
    print type(filename)
    print filename.split('.')[0]

    img_name_jgp = img_path + "\\" +filename.split('.')[0]+".jpg"
    img_name_JPG = img_path + "\\" +filename.split('.')[0]+".JPG"

    
    if(os.path.exists(img_name_jgp)):
       img_save_name = img_save_path + '\\' + filename.split('.')[0]+'.jpg'
       img = cv2.imread(img_name_jgp)
       cv2.imwrite(img_save_name,img)

    if(os.path.exists(img_name_JPG)):
       img_save_name = img_save_path + '\\' + filename.split('.')[0]+'.JPG'
       img = cv2.imread(img_name_JPG)
       cv2.imwrite(img_save_name,img)
'''       
 

#img_path = 'E:\\Image\\shandong\\primary_sample_xml\\Image'
#xml_path = 'E:\\Image\\shandong\\primary_sample_xml\\primary_sample_xml'
    


def create_train_dateset(xml_path,train_save_path,train_val_save_path,test_save_path,val_save_path):
    
    list = os.listdir(xml_path)
    length = len(list)
    
    train_val_num = int(length*0.8)
    train_num = int(train_val_num*0.75)
    val_num = int(train_val_num*0.25)
    test_num = int(length*0.2)
    
    rand_index = random.sample(list,length)
    train_val_index = rand_index[0:train_val_num]
    val_index = rand_index[train_num:train_val_num]
    train_index = rand_index[0:train_num]
    test_index = rand_index[train_val_num:length]
    
    with open(train_save_path,'w+') as file_object:
        for xml_file in train_index:
            xml_name = xml_file.split('.')[0]
            file_object.write(xml_name)
            file_object.write('\n')
    
    with open(train_val_save_path,'w+') as file_object:
        for xml_file in train_val_index:
            xml_name = xml_file.split('.')[0]
            file_object.write(xml_name)
            file_object.write('\n')


    with open(test_save_path,'w+') as file_object:
        for xml_file in test_index:
            xml_name = xml_file.split('.')[0]
            file_object.write(xml_name)
            file_object.write('\n')

    with open(val_save_path,'w+') as file_object:
        for xml_file in val_index:
            xml_name = xml_file.split('.')[0]
            file_object.write(xml_name)
            file_object.write('\n')


def main():
	#xml_path = 'E:/Image/'		#原xml路径
	#jpg_path = 'D:/wuhandata/dachicun/jpg/'				#原jpg路径
	#take_jpg_path = 'C:/Users/admin/Desktop/int/xml/'	#提取出的jpg保存的路径
    	
	#new_xml_path = 'F:/' #保存xml的路径	
	
	#1、提取与xml同名的jpg图片
	#take_jpg(xml_path,jpg_path,take_jpg_path)
	
	#2、提取只包含 line_insulator 标签的xml到文件夹
	# need_xml_path = new_xml_path
	 #take_want_bbox_xml(xml_path,need_xml_path)
	
	#3、提取xml中使用的类别
	#del_object(xml_path,new_xml_path)

    #4.提取训练、测试文件目录
    xml_path = '/home/ly/data/VOC2012/Annotations'
    train_save_path = '/home/ly/data/VOC2012/ImageSets/Main/train.txt'
    train_val_save_path = '/home/ly/data/VOC2012/ImageSets/Main/trainval.txt'
    test_save_path = '/home/ly/data/VOC2012/ImageSets/Main/test.txt'
    val_save_path =  '/home/ly/data/VOC2012/ImageSets/Main/val.txt'

    create_train_dateset(xml_path,train_save_path,train_val_save_path,test_save_path,val_save_path)
	
main()

