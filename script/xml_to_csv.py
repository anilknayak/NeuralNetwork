import numpy as np
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import sys
import glob
from tqdm import tqdm

def main(read_images_base_dir,csv_write_dir):
    for directory in tqdm(['train','test'], desc = "Converting XML to CSV ", ncols = 100):
        image_path = os.path.join(os.getcwd(), read_images_base_dir+'images/{}'.format(directory))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(csv_write_dir+'data/{}_labels.csv'.format(directory), index=None)
        # print('Successfully converted xml to csv. for ',directory,'ing')

def xml_to_csv(path):
    xml_list = []
    for xml_file in tqdm(glob.glob(path + '/*.xml'),desc = "Converting", ncols = 100):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        fileName = root.find('filename').text
        size_el = root.find('size')
        width = size_el.find('width').text
        height = size_el.find('height').text

        obj = root.findall('object')[0]
        class_label = obj.find('name').text
        boundingbox = obj.find('bndbox')
        xmin = boundingbox.find('xmin').text
        ymin = boundingbox.find('ymin').text
        xmax = boundingbox.find('xmax').text
        ymax = boundingbox.find('ymax').text


        value = (fileName,width,height,class_label,xmin,ymin,xmax,ymax)
        xml_list.append(value)

        # for member in root.findall('object'):
        #     value = (root.find('filename').text,
        #              int(root.find('size')[0].text),
        #              int(root.find('size')[1].text),
        #              member[0].text,
        #              int(member[4][0].text),
        #              int(member[4][1].text),
        #              int(member[4][2].text),
        #              int(member[4][3].text)
        #              )
        #     xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

number_of_arg = len(sys.argv)
read_path = sys.argv[1]
write_path = sys.argv[2]

if number_of_arg==3:
    main(read_path,write_path)
    # print("==========================================================================================")
else:
    print("Lesser Number of argument")
    print("arg1 : read path")
    print("arg2 : write path")
