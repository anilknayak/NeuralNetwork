
# Created by anilnayak
# Creation Information
# Date 10/22/17 
# Year 2017 
# Time 5:15 AM
# For the Project TensorFlow
# created from PyCharm Community Edition

from pathlib import Path
import os
import xml.etree.ElementTree as etree
import shutil
import sys
import glob

number_of_arg = len(sys.argv)
annotation_path = sys.argv[1]
images_path = sys.argv[2]
target_base_path = sys.argv[3]

# base_path = "/home/anil/training_details/training_images"
# target_base_path = "/home/anil/training_details/training_images/dataset/images"

# base_path = "/Users/anilnayak/Downloads/trained/NeuralNetwork/Attempt3"
# target_base_path = "/Users/anilnayak/Downloads/trained/NeuralNetwork/Attempt3/images"

# annotation_path = base_path+"/VOCdevkit/VOC2012/Annotations"
# images_path = base_path+"/VOCdevkit/VOC2012/JPEGImages"

is_exist_ann = os.path.exists(annotation_path)
is_exist_img = os.path.exists(images_path)

if number_of_arg == 4:

    category = {}

    if is_exist_ann and is_exist_img:
        print("Listing the Images and Annotations Available")
        annotations_paths = glob.glob(annotation_path+"/*.xml")
        # annotations_paths = os.listdir(annotation_path)
        print("Number of Annotation Available : ",str(len(annotations_paths)))
        # images_paths = os.listdir(images_path)
        # print("Number of Images Available : ", str(len(images_paths)))

        categories = {}
        for annotation_file_path in annotations_paths:
            annotation_file_path = annotation_file_path
            tree = etree.parse(annotation_file_path)
            root = tree.getroot()

            file_info = ''
            dtl = {}
            dtl['annotation'] = annotation_file_path
            for child in root:
                if child.tag == 'object':
                    for child_l_2 in child:
                        if child_l_2.tag == 'name':
                            file_info = child_l_2.text
                            break

                if child.tag == 'filename':
                    images_path_filename = images_path+"/"+child.text
                    dtl['image'] = images_path_filename

            if file_info not in categories:
                file_list = []
                file_list.append(dtl)
                categories[file_info] = file_list
            else:
                categories[file_info].append(dtl)

        distributions = {}
        for category in categories.keys():
            list_of_files = categories[category]
            total = len(list_of_files)
            testing = int((total/100)*5)

            inv_cat_dtl = {}
            inv_cat_dtl['total'] = total
            inv_cat_dtl['training'] = total-testing
            inv_cat_dtl['testing'] = testing
            inv_cat_dtl['files'] = list_of_files

            distributions[category] = inv_cat_dtl

        sum = 0
        sum1 = 0
        sum2 = 0
        for distribution in distributions.keys():
            dis = distributions[distribution]
            files = dis['files']
            print("Category : [",distribution , "] \t\t => Total : " , dis['total'] , " Training : ", dis['training'] , " Testing : ", dis['testing'])

            training_init = int(dis['training'])
            sum = sum + int(dis['total'])
            sum1 = sum1 + training_init
            sum2 = sum2 + int(dis['testing'])

            print("Preparing Data Distribution for Class Label : ",distribution)

            training_count = 0
            training_flag = True
            for file_dict in files:
                annotation_file = file_dict['annotation']
                image_file = file_dict['image']
                training_count = training_count + 1

                if training_flag:
                    shutil.copy2(annotation_file,target_base_path+"/train/")
                    shutil.copy2(image_file, target_base_path+"/train/")
                else:
                    shutil.copy2(annotation_file, target_base_path+"/test/")
                    shutil.copy2(image_file, target_base_path+"/test/")

                shutil.copy2(annotation_file, target_base_path+"/")
                shutil.copy2(image_file, target_base_path+"/")

                if training_count==training_init:
                    training_flag = False

        if sum == (sum1+sum2):
            print("Total Samples : ",sum, " Training Samples: ",sum1," Tesrting Samples: ",sum2)
    else:
        print("Either Image or Annotation path does not exist")
else:
    print("Lesser number of argument passes")
    print("arg1 : path of the annotation directory")
    print("arg2 : path to the target directory for the training and testing segregation")