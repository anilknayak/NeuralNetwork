
rm -R /home/anil/training_details/training/images/
mkdir /home/anil/training_details/training/images
chmod 777 /home/anil/training_details/training/images
mkdir /home/anil/training_details/training/images/train
chmod 777 /home/anil/training_details/training/images/train
mkdir /home/anil/training_details/training/images/test
rm /home/anil/models/research/object_detection/data/*
rm -R /home/anil/models/research/object_detection/images/
rm /home/anil/models/research/object_detection/training/*
rm /home/anil/training_details/training/data/*
rm /home/anil/training_details/training/training/*
rm -R /home/anil/models/research/object_detection/ssd_mobilenet_v1_coco_11_06_2017/

cp -a /home/anil/training_details/training/model/object-detection.pbtxt /home/anil/training_details/training/data/
cp -a /home/anil/training_details/training/model/ssd_mobilenet_v1_coco.config /home/anil/training_details/training/data/
cp -a /home/anil/training_details/training/model/ssd_mobilenet_v1_coco.config /home/anil/training_details/training/training/

python SplitTrainingTestingData.py /home/anil/training_details/training_images/VOCdevkit/VOC2012/Annotations /home/anil/training_details/training_images/VOCdevkit/VOC2012/JPEGImages /home/anil/training_details/training/images
python SplitTrainingTestingData.py /home/anil/training_details/training_images/dataset /home/anil/training_details/training_images/dataset /home/anil/training_details/training/images

python xml_to_csv.py /home/anil/training_details/training/ /home/anil/training_details/training/

python generate_tfrecord.py --csv_input=/home/anil/training_details/training/data/train_labels.csv --output_path=/home/anil/training_details/training/data/train.record --image_path=/home/anil/training_details/training/
python generate_tfrecord.py --csv_input=/home/anil/training_details/training/data/test_labels.csv --output_path=/home/anil/training_details/training/data/test.record --image_path=/home/anil/training_details/training/

cp -a /home/anil/training_details/training/data/* /home/anil/models/research/object_detection/data/
cp -a /home/anil/training_details/training/images/ /home/anil/models/research/object_detection/
cp -a /home/anil/training_details/training/training/* /home/anil/models/research/object_detection/training/
cp -a /home/anil/training_details/training/model/ssd_mobilenet_v1_coco_11_06_2017/ /home/anil/models/research/object_detection/


cd /home/anil/models/research/ && export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

cd /home/anil/models/research/object_detection/ && python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config








