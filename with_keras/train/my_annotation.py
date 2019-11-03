#%%
import os
import xml.etree.ElementTree as ET
from os import getcwd

classes = ["stapler","ZhuPai"]

filePath = 'OpenLabeling/main/input'
files = os.listdir(filePath)
files.sort()
sets=[]
for image in files:
    (filename, extension) = os.path.splitext(image)
    sets.append(filename)

def convert_annotation(image_id, list_file):
    in_file = open('OpenLabeling/main/output/PASCAL_VOC/%s.xml'%(image_id))
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

wd = getcwd()

list_file = open('Stapler_ZhuPai.txt', 'w')
for image_id in sets:
    list_file.write('%s/OpenLabeling/main/input/%s.jpg'%(wd, image_id))
    convert_annotation(image_id, list_file)
    list_file.write('\n')
list_file.close()

# %%
