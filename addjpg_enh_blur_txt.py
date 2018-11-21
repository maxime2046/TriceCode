#-*-coding:utf-8-*-

import os
import cv2
import numpy as np
import shutil
import random
from PIL import Image
from PIL import ImageEnhance


# 1、单纯图片预处理操作
class ImgRotation(object):
    def __init__(self):
        pass
    # 1、图像旋转
    def rotation(self, image, center_x, center_y, angle):
        M = cv2.getRotationMatrix2D((center_x,center_y),angle,1)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return rotated    # 返回转换后的图像

    def img_to_rotate(self,old_jpg_path,new_jpg_path,angle,name_rotate):
        # 读取注释文件
        #for xml_file in glob.glob(path + '/*.xml'):
        for jpg_name in os.listdir(old_jpg_path):
            try:
                img = cv2.imread(old_jpg_path+jpg_name)
                print(jpg_name)
                #img_show = cv2.resize(img,(1200,800))
                #cv2.imshow('source'+jpg_name, img_show)
                #cv2.waitKey(1)
            except:
                info = 'img_to_defog fail to read %s' % (jpg_name)
                print(info)
                break
            # 将原图按照某点旋转多少度,此处选中心点，angle通常选取的小点，10度啥的
            center_x = int(img.shape[0] / 2)
            center_y = int(img.shape[1] / 2)
            rotate_img = self.rotation(img,center_x,center_y,angle)
            #rotate_img = cv2.resize(rotate_img,(1072,712))

            name = jpg_name.split('.')
            img_name = name[0]
            cv2.imwrite(new_jpg_path + img_name + name_rotate +'.jpg',rotate_img)
            #rotate_show = cv2.resize(rotate_img, (1200, 800))
            #cv2.imshow('rotate' + jpg_name, rotate_show)
            #cv2.waitKey(1)
        return 0


# 2、单纯图片去雾操作 何凯明 https://www.cnblogs.com/Imageshop/p/3281703.html
class ImgHazeRemoval(object):
    def __init__(self):
        pass

    def zmMinFilterGray(self, src, r=11):
        '''最小值滤波，r是滤波器半径'''
        if r <= 0:
            return src
        h, w = src.shape[:2]
        I = src
        res = np.minimum(I, I[[0] + [x for x in range(h-1)], :])
        res = np.minimum(res, I[[x for x in range(1,h)] + [h - 1], :])
        I = res
        res = np.minimum(I, I[:, [0] + [x for x in range(w-1)]])
        res = np.minimum(res, I[:, [x for x in range(1,w)] + [w - 1]])
        return self.zmMinFilterGray(res, r - 1)

    def guidedfilter(self, I, p, r, eps):
        '''引导滤波，直接参考网上的matlab代码'''
        height, width = I.shape
        m_I = cv2.boxFilter(I, -1, (r, r))
        m_p = cv2.boxFilter(p, -1, (r, r))
        m_Ip = cv2.boxFilter(I * p, -1, (r, r))
        cov_Ip = m_Ip - m_I * m_p

        m_II = cv2.boxFilter(I * I, -1, (r, r))
        var_I = m_II - m_I * m_I

        a = cov_Ip / (var_I + eps)
        b = m_p - a * m_I

        m_a = cv2.boxFilter(a, -1, (r, r))
        m_b = cv2.boxFilter(b, -1, (r, r))
        return m_a * I + m_b

    def getV1(self, m, r, eps, w, maxV1):  # 输入rgb图像，值范围[0,1]
        '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
        V1 = np.min(m, 2)  # 得到暗通道图像
        V1 = self.guidedfilter(V1, self.zmMinFilterGray(V1, 7), r, eps)  # 使用引导滤波优化
        bins = 2000
        ht = np.histogram(V1, bins)  # 计算大气光照A
        d = np.cumsum(ht[0]) / float(V1.size)
        for lmax in range(bins - 1, 0, -1):
            if d[lmax] <= 0.999:
                break
        A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()

        V1 = np.minimum(V1 * w, maxV1)  # 对值范围进行限制
        return V1, A

    def deHaze(self, m, r=25, eps=0.001, w=0.8, maxV1=0.80, bGamma=False):#r=101
        Y = np.zeros(m.shape)
        V1, A = self.getV1(m, r, eps, w, maxV1)  # 得到遮罩图像和大气光照
        for k in range(3):
            Y[:, :, k] = (m[:, :, k] - V1) / (1 - V1 / A)  # 颜色校正
        Y = np.clip(Y, 0, 1)
        if bGamma:
            Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作
        return Y

    def img_to_defog(self,old_jpg_path,new_jpg_path,name_defog):
        for jpg_name in os.listdir(old_jpg_path):
            try:
                img = cv2.imread(old_jpg_path+jpg_name)
                #print(jpg_name)
                #print(img.shape[0])
                #img_show = cv2.resize(img,(1200,800))
                #cv2.imshow('source'+jpg_name, img_show)
                #cv2.waitKey(1)
            except:
                info = 'img_to_defog fail to read %s' % (jpg_name)
                print(info)
                break
            print(type(img))
            defog_img = self.deHaze(img / 255.0) * 255
            #(type(defog_img[1,1,1]))
            defog_img = defog_img.astype(np.uint8)  #numpy矩阵中元素从 numpy.float64格式转为 numpy.uint8
            name = jpg_name.split('.')
            img_name = name[0]
            cv2.imwrite(new_jpg_path + img_name + name_defog + '.jpg', defog_img)
            #cv2.imshow('defog.jpg', defog_img)
            #cv2.waitKey()
        return 0


# 3、图像镜像
def ImgFlip(old_jpg_path,new_jpg_path,name_flip):
    for jpg_name in os.listdir(old_jpg_path):
        try:
            img = cv2.imread(old_jpg_path + jpg_name)
            #print(jpg_name)
            #img_show = cv2.resize(img, (1200, 800))
            #cv2.imshow('source' + jpg_name, img_show)
            #cv2.waitKey(1)
        except:
            info = 'ImgFlip fail to read %s' % (jpg_name)
            print(info)
            break
        flip_img = cv2.flip(img, 1, dst=None)  # 水平镜像
        #flip_img = cv2.flip(img, 0, dst=None)  # 垂直镜像
        #flip_img = cv2.flip(img, -1, dst=None)  # 对角镜像
        name = jpg_name.split('.')
        img_name = name[0]
        cv2.imwrite(new_jpg_path + img_name + name_flip + '.jpg', flip_img)

class ImgEnhance(object):
    def __init__(self):
        pass

    # 1、亮度增强
    def image_brighten(self, image, new_name):
        enh_bri = ImageEnhance.Brightness(image)
        brightness = 1.2#1.5
        image_brightened = enh_bri.enhance(brightness)
        # image_brightened.show()
        image_brightened = cv2.cvtColor(np.asarray(image_brightened), cv2.COLOR_RGB2BGR)  # PIL.Image转换成OpenCV格式
        #image_brightened = cv2.resize(image_brightened, (1200, 800))
        #print(new_name)
        cv2.imwrite(new_name, image_brightened)
        #cv2.imshow("image_brightened", image_brightened)
        #cv2.waitKey()

    # 2、色度增强
    def image_colored(self, image,new_name):
        enh_col = ImageEnhance.Color(image)
        color = 1.5
        image_colored = enh_col.enhance(color)
        #image_colored.show()
        image_colored = cv2.cvtColor(np.asarray(image_colored), cv2.COLOR_RGB2BGR)  # PIL.Image转换成OpenCV格式
        #image_colored = cv2.resize(image_colored, (1200, 800))
        cv2.imwrite(new_name, image_colored)
        #cv2.imshow("image_colored", image_colored)
        #cv2.waitKey()

    # 3、对比度增强
    def image_contrast(self, image,new_name):
        enh_con = ImageEnhance.Contrast(image)
        contrast = 1.5
        image_contrasted = enh_con.enhance(contrast)
        #image_contrasted.show()
        image_contrasted = cv2.cvtColor(np.asarray(image_contrasted), cv2.COLOR_RGB2BGR)  # PIL.Image转换成OpenCV格式
        #image_contrasted = cv2.resize(image_contrasted, (1200, 800))
        cv2.imwrite(new_name, image_contrasted)
        #cv2.imshow("image_contrasted", image_contrasted)
        #cv2.waitKey()


    # 4、锐度增强
    def image_sharp(self, image,new_name):
        enh_sha = ImageEnhance.Sharpness(image)
        sharpness = 3.0
        image_sharped = enh_sha.enhance(sharpness)
        #image_sharped.show()
        image_sharped = cv2.cvtColor(np.asarray(image_sharped), cv2.COLOR_RGB2BGR)  # PIL.Image转换成OpenCV格式
        #image_sharped = cv2.resize(image_sharped, (1200, 800))
        cv2.imwrite(new_name, image_sharped)
        #cv2.imshow("image_sharped", image_sharped)
        #cv2.waitKey()


    def image_to_Enhance(self, old_jpg_path, new_jpg_path,old_xml_path , new_xml_path, name_enhance):
        for xml_name in os.listdir(old_xml_path):
            #img = np.zeros((256, 256, 3), dtype='uint8')  # 初始三通道
            #img = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8))  # RGB image
            #try:
                name = xml_name.split('.')
                img_name = name[0]
                img = Image.open(old_jpg_path + img_name + '.jpg')
                #print(old_jpg_path + img_name + '.JPG')
                #new_name = new_jpg_path + img_name + name_enhance + '.JPG'
                # 可四选一 image_brighten、image_colored、image_sharp、image_to_Enhance
                a = random.randint(1,12) #随机整数
                name1 = old_xml_path+img_name+'.xml'
                #print(a)
                if 4 > a >= 1:  #1、2、3
                    self.image_brighten(img, new_jpg_path + img_name + name_enhance +'brighten'+ '.jpg')   #1
                    name2 = new_xml_path + img_name + name_enhance +'brighten'+ '.xml'
                    shutil.copyfile(name1,name2)
                elif 7> a >= 4: #4、5、6
                    self.image_colored(img, new_jpg_path + img_name + name_enhance +'colored'+ '.jpg')     #2
                    name2 = new_xml_path + img_name + name_enhance +'colored'+ '.xml'
                    shutil.copyfile(name1,name2)
                elif 10> a >= 7: #7、8、9    
                    self.image_contrast(img, new_jpg_path + img_name + name_enhance +'contrast'+ '.jpg')    #3
                    name2 = new_xml_path + img_name + name_enhance +'contrast'+ '.xml'
                    shutil.copyfile(name1,name2)
                elif 13> a >= 10: #10、11、12      
                    self.image_sharp(img, new_jpg_path + img_name + name_enhance +'sharp'+ '.jpg')         #4
                    name2 = new_xml_path + img_name + name_enhance +'sharp'+ '.xml'
                    shutil.copyfile(name1,name2)
                
           # except:
                #info = 'ImgEnhance fail to read %s' % (xml_name)
               # print(info)
                #break
    def image_to_Enhance_txt(self, old_jpg_path, new_jpg_path,old_xml_path , new_xml_path, name_enhance,txt_file):
        #for xml_name in os.listdir(old_xml_path):
        file = open(txt_file)
        for line in file:
                img_name = line[:len(line)-1]
                img = Image.open(old_jpg_path + img_name + '.jpg')
                #print(old_jpg_path + img_name + '.JPG')
                #new_name = new_jpg_path + img_name + name_enhance + '.JPG'
                # 可四选一 image_brighten、image_colored、image_sharp、image_to_Enhance
                a = random.randint(1,12) #随机整数
                name1 = old_xml_path+img_name+'.xml'
                #print(a)
                if 4 > a >= 1:  #1、2、3
                    self.image_brighten(img, new_jpg_path + img_name + name_enhance +'brighten'+ '.jpg')   #1
                    name2 = new_xml_path + img_name + name_enhance +'brighten'+ '.xml'
                    shutil.copyfile(name1,name2)
                elif 7> a >= 4: #4、5、6
                    self.image_colored(img, new_jpg_path + img_name + name_enhance +'colored'+ '.jpg')     #2
                    name2 = new_xml_path + img_name + name_enhance +'colored'+ '.xml'
                    shutil.copyfile(name1,name2)
                elif 10> a >= 7: #7、8、9    
                    self.image_contrast(img, new_jpg_path + img_name + name_enhance +'contrast'+ '.jpg')    #3
                    name2 = new_xml_path + img_name + name_enhance +'contrast'+ '.xml'
                    shutil.copyfile(name1,name2)
                elif 13> a >= 10: #10、11、12      
                    self.image_sharp(img, new_jpg_path + img_name + name_enhance +'sharp'+ '.jpg')         #4
                    name2 = new_xml_path + img_name + name_enhance +'sharp'+ '.xml'
                    shutil.copyfile(name1,name2)
                
           # except:
                #info = 'ImgEnhance fail to read %s' % (xml_name)
               # print(info)
                #break

class ImgBlur(object):
#随机选择四种模糊方式：
    def __init__(self):
        pass
#1 、均值模糊
    def mean_blur(self, image,new_name):
        #参数（5，5）：表示高斯矩阵的长与宽都是5
        dst = cv2.blur(image,(5,5))
        cv2.imwrite(new_name, dst)
        #cv2.imshow("mean blur",dst)
        #cv2.waitKey()
#2、中值模糊
    def mid_blur(self, image,new_name):                         
        #第二个参数是孔径的尺寸，一个大于1的奇数。比如这里是5，中值滤波器就会使用5×5的范围来计算。即对像素的中心值及其5×5邻域组成了一个数值集，对其进行处理计算，当前像素被其中值替换掉。
        dst = cv2.medianBlur(image, 5)
        cv2.imwrite(new_name, dst)
        #cv.imshow("median", dst)

#3、高斯模糊
    def gaus_blur(self, image,new_name):
        #这里(5, 5)表示高斯矩阵的长与宽都是5，标准差取0时OpenCV会根据高斯矩阵的尺寸自己计算。
        dst = cv2.GaussianBlur(image,(5,5),0)
        cv2.imwrite(new_name, dst)
        
#4、自定义模糊
    def custom_blur(self, image,new_name):
        #定义一个5*5的卷积核
        kernel = np.ones([5,5],np.float32)/25
        dst = cv2.filter2D(image,-1,kernel=kernel)
        cv2.imwrite(new_name, dst)
        #cv.imshow("custom", dst)

    def image_to_blur(self, old_jpg_path, new_jpg_path,old_xml_path , new_xml_path, name_blur):    
        for xml_name in os.listdir(old_xml_path):
            try:
                name = xml_name.split('.')
                img_name = name[0]
                img = cv2.imread(old_jpg_path + img_name + '.jpg')
                #print(old_jpg_path + img_name + '.JPG')
                name1 = old_xml_path+img_name+'.xml'
                a = random.randint(1,12) #随机整数
                #print(a)
                if 4 > a >= 1:  #1、2、3
                    self.mean_blur(img, new_jpg_path + img_name + name_blur +'mean_blur'+ '.jpg')  #1
                    name2 = new_xml_path + img_name + name_blur +'mean_blur'+ '.xml'
                    shutil.copyfile(name1,name2)
                elif 7> a >= 4: #4、5、6
                    self.mid_blur(img, new_jpg_path + img_name + name_blur +'mid_blur'+ '.jpg')   #2
                    name2 = new_xml_path + img_name + name_blur +'mid_blur'+ '.xml'
                    shutil.copyfile(name1,name2)
                elif 10> a >= 7: #7、8、9
                    self.gaus_blur(img, new_jpg_path + img_name + name_blur +'gaus_blur'+ '.jpg')   #3
                    name2 = new_xml_path + img_name + name_blur +'gaus_blur'+ '.xml'
                    shutil.copyfile(name1,name2)
                elif 13> a >= 10: #10、11、12 
                    self.custom_blur(img, new_jpg_path + img_name + name_blur +'custom'+ '.jpg')     #4 
                    name2 = new_xml_path + img_name + name_blur +'custom'+ '.xml'
                    shutil.copyfile(name1,name2)
            except:
                info = 'ImgBlur fail to read %s' % (xml_name)
                print(info) 
    
    def image_to_blur_txt(self, old_jpg_path, new_jpg_path,old_xml_path , new_xml_path, name_blur,txt_file):    
        #for xml_name in os.listdir(old_xml_path):
        file = open(txt_file)
        for line in file:
            try:
                img_name = line[:len(line)-1]
                img = cv2.imread(old_jpg_path + img_name + '.jpg')
                #print(old_jpg_path + img_name + '.JPG')
                name1 = old_xml_path+img_name+'.xml'
                a = random.randint(1,12) #随机整数
                #print(a)
                if 4 > a >= 1:  #1、2、3
                    self.mean_blur(img, new_jpg_path + img_name + name_blur +'mean_blur'+ '.jpg')  #1
                    name2 = new_xml_path + img_name + name_blur +'mean_blur'+ '.xml'
                    shutil.copyfile(name1,name2)
                elif 7> a >= 4: #4、5、6
                    self.mid_blur(img, new_jpg_path + img_name + name_blur +'mid_blur'+ '.jpg')   #2
                    name2 = new_xml_path + img_name + name_blur +'mid_blur'+ '.xml'
                    shutil.copyfile(name1,name2)
                elif 10> a >= 7: #7、8、9
                    self.gaus_blur(img, new_jpg_path + img_name + name_blur +'gaus_blur'+ '.jpg')   #3
                    name2 = new_xml_path + img_name + name_blur +'gaus_blur'+ '.xml'
                    shutil.copyfile(name1,name2)
                elif 13> a >= 10: #10、11、12 
                    self.custom_blur(img, new_jpg_path + img_name + name_blur +'custom'+ '.jpg')     #4 
                    name2 = new_xml_path + img_name + name_blur +'custom'+ '.xml'
                    shutil.copyfile(name1,name2)
            except:
                info = 'ImgBlur fail to read %s' % (xml_name)
                print(info) 
     
                
                
def main():
 
    txt_file = '/home/ly/project/incubator-mxnet/example/ssd/data/VOCdevkit/VOC2007/ImageSets/Main/train.txt'
    old_jpg_path = '/home/ly/project/incubator-mxnet/example/ssd/data/VOCdevkit/VOC2007/JPEGImages/'   #'D:/devel/python_test/img/JPG/'
    new_jpg_path = '/home/ly/project/incubator-mxnet/example/ssd/data/VOCdevkit/VOC2007/JPEGImages/'    #'D:/devel/python_test/img/new_JPG/'
    
    old_xml_path = '/home/ly/project/incubator-mxnet/example/ssd/data/VOCdevkit/VOC2007/Annotations/'    
    new_xml_path = '/home/ly/project/incubator-mxnet/example/ssd/data/VOCdevkit/VOC2007/Annotations/'    
    
    #txt_file = '/home/ly/project/incubator-mxnet/example/ssd/data/VOCdevkit/VOC2007/ImageSets/Main/val.txt'
    #old_jpg_path = '/home/ly/project/incubator-mxnet/example/ssd/data/VOCdevkit/VOC2007/JPEGImages/'   #'D:/devel/python_test/img/JPG/'
    #new_jpg_path = '/home/ly/project/incubator-mxnet/example/ssd/data/VOCdevkit/VOC2007/JPEGImages/'    #'D:/devel/python_test/img/new_JPG/'
    
    #old_xml_path = '/home/ly/project/incubator-mxnet/example/ssd/data/VOCdevkit/VOC2007/Annotations/'    
    #new_xml_path = '/home/ly/project/incubator-mxnet/example/ssd/data/VOCdevkit/VOC2007/Annotations/'
    
    index_rotation = False#True#扩充样本用不到
    index_defog = False#True#扩充样本用不到
    index_flip = False#True# 扩充样本用不到

    index_enhance =  True#False
    index_blur =  True#True

    # 1、旋转
    if index_rotation:
        angle = 5
        name_rotate = 'rotate'+str(angle)
        imgRotation = ImgRotation()
        imgRotation.img_to_rotate(old_jpg_path,new_jpg_path,angle,name_rotate)

    # 2、除雾
    if index_defog:
        #img = cv2.imread('D:/devel/python_test/img/JPG/fog.JPG')
        #print(img.shape[2])
        name_defog = 'defog'
        imgHazeRemoval =ImgHazeRemoval()
        imgHazeRemoval.img_to_defog(old_jpg_path,new_jpg_path,name_defog)

    # 3、镜像
    # python 图像翻转,使用openCV flip()方法翻转
    name_flip = 'flip'
    if index_flip:
        ImgFlip(old_jpg_path, new_jpg_path, name_flip)
        #img = cv2.imread('D:/devel/python_test/img/JPG/fog.JPG')
        #xImg = cv2.flip(img, 1, dst=None)  # 水平镜像
        #cv2.imshow('defog.jpg', xImg)
        #cv2.waitKey(1)
        #cv2.imwrite(new_jpg_path + 'defog.jpg', xImg)

    # 4、图像增强
    if index_enhance:
        # 原始图像
        name_enhance = 'enhance'
        imgEnhance = ImgEnhance()
        #imgEnhance.image_to_Enhance(old_jpg_path,new_jpg_path,old_xml_path,new_xml_path,name_enhance)
        imgEnhance.image_to_Enhance_txt(old_jpg_path,new_jpg_path,old_xml_path,new_xml_path,name_enhance,txt_file)
        #image = Image.open('D:/devel/python_test/img/JPG/1522809427119dachicun.JPG')
        #image.show()

    # 5、图像模糊
    if index_blur:
       name_blur = 'blur'
       imgBlur = ImgBlur()
       #imgBlur.image_to_blur(old_jpg_path,new_jpg_path,old_xml_path,new_xml_path,name_blur)
       imgBlur.image_to_blur_txt(old_jpg_path,new_jpg_path,old_xml_path,new_xml_path,name_blur,txt_file)

if __name__ == '__main__':
    main()
