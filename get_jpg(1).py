# coding=utf-8

#1 ����take_jpg����ȡ��xmlͬ����jpg
#2 ����del_object����ȡxml��ʹ�õ����(ɾ�����õ����)

import os
import shutil
import xml.etree.ElementTree as ET

def take_jpg(xml_path,jpg_path,take_jpg_path):
#1 ��ȡxmlͬ����jpg
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
#2 ��ȡֻ���� line_insulator ��ǩ��xml���ļ���
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
#3 ��ȡxml��ʹ�õ����

	for xml_file_name in os.listdir(xml_path):
		xml_file_ = xml_path + xml_file_name
		
		#print(xml_file_)
		tree = ET.parse(xml_file_)
		root = tree.getroot()
		
		#������Ҫ�����,����ɾ�������object
		for member in root.findall('object'):
			if member[0].text != 'line_insulator':
				root.remove(member)
			
		tree.write(new_xml_path + xml_file_name)	
	
	return 0
	
	
def main():
	xml_path = '/home1/data/yu/jyzzb/Annotations/'		#ԭxml·��
	jpg_path = '/home1/data/yu/jyz_jpg/'				#ԭjpg·��
	take_jpg_path = '/home1/data/yu/jyzzb/JPEGImages'	#��ȡ����jpg�����·��
	
	#new_xml_path = 'C:/Users/admin/Desktop/int/new_xml/' #����xml��·��
	
	
	#1����ȡ��xmlͬ����jpgͼƬ
	take_jpg(xml_path,jpg_path,take_jpg_path)
	
	#2����ȡֻ���� line_insulator ��ǩ��xml���ļ���
	# need_xml_path = new_xml_path
	# take_want_bbox_xml(xml_path,need_xml_path)
	
	#3����ȡxml��ʹ�õ����
	#del_object(xml_path,new_xml_path)
	
main()
