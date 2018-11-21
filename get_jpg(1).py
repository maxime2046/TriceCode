# coding=utf-8

#1 函数take_jpg，提取与xml同名的jpg
#2 函数del_object，提取xml中使用的类别(删除不用的类别)

import os
import shutil
import xml.etree.ElementTree as ET

def take_jpg(xml_path,jpg_path,take_jpg_path):
#1 提取xml同名的jpg
#
	for xml_file_name in os.listdir(xml_path):
		xml_file = xml_path + xml_file_name
		xml_name = xml_file_name.split('.') 
		only_name_xml = xml_name[0]
		for jpg_file_name in os.listdir(jpg_path):
			jpg_xml = jpg_file_name.split('.')
			only_name_jpg = jpg_xml[0]
			if only_name_jpg == only_name_xml:
				old_path = jpg_path
				new_path = take_jpg_path
				file_name = only_name_jpg +'.JPG'
				shutil.copyfile(os.path.join(old_path,file_name),os.path.join(new_path,file_name))
	return 0

	
def take_want_bbox_xml(xml_path,need_xml_path):
#2 提取只包含 line_insulator 标签的xml到文件夹
	for xml_file_name in os.listdir(xml_path):
		xml_file = xml_path + xml_file_name
		print(xml_file)
		tree = ET.parse(xml_file)
		root = tree.getroot()
		index = False
		for member in root.findall('object'):
			if member[0].text == 'line_insulator':
				index = True
		
		if index == True:
			shutil.copyfile(xml_file,os.path.join(need_xml_path,xml_file_name))
	return 0

	
def del_object(xml_path,new_xml_path):
#3 提取xml中使用的类别

	for xml_file_name in os.listdir(xml_path):
		xml_file_ = xml_path + xml_file_name
		
		#print(xml_file_)
		tree = ET.parse(xml_file_)
		root = tree.getroot()
		
		#查找需要的类别,否则删除此类别object
		for member in root.findall('object'):
			if member[0].text != 'line_insulator':
				root.remove(member)
			
		tree.write(new_xml_path + xml_file_name)	
	
	return 0
	
	
def main():
	xml_path = '/home1/data/yu/jyzzb/Annotations/'		#原xml路径
	jpg_path = '/home1/data/yu/jyz_jpg/'				#原jpg路径
	take_jpg_path = '/home1/data/yu/jyzzb/JPEGImages'	#提取出的jpg保存的路径
	
	#new_xml_path = 'C:/Users/admin/Desktop/int/new_xml/' #保存xml的路径
	
	
	#1、提取与xml同名的jpg图片
	take_jpg(xml_path,jpg_path,take_jpg_path)
	
	#2、提取只包含 line_insulator 标签的xml到文件夹
	# need_xml_path = new_xml_path
	# take_want_bbox_xml(xml_path,need_xml_path)
	
	#3、提取xml中使用的类别
	#del_object(xml_path,new_xml_path)
	
main()
