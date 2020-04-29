import os
import xml.etree.ElementTree as ET
import tensorflow as tf
from object_detection.utils import dataset_util



# This are the path to the datasets and to the output files.
# NEED TO BE UPDATED IN CASE THE DATASET CHANGES
PATH_TEST = "2361allDataset/test/"
PATH_RECORD_TEST = "newtest.record"
PATH_TRAIN = "2361allDataset/train/"
PATH_RECORD_TRAIN = "newtrain.record"

IMAGE_EXT = ".jpg"
IMAGE_FORMAT = b'jpg'

# This function defines the different classes the dataset has and return a different number per each.
# NEED TO BE UPDATED IN CASE THE DATASET CHANGES
def class_text_to_int(row_label):
    if row_label == 'nut':
        return 1
    if row_label == 'bolt':
        return 2
    else:
        return 3

# Reads the xml and the images, and create the tf records files. 
def xml_to_tf(path_input, path_output):
    xml_list = []
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    writer = tf.compat.v1.python_io.TFRecordWriter(path_output)

    files = os.listdir(path_input)
    for file in files:
        if file.endswith(".xml"):
            xmlFile = path_input + file

            tree = ET.parse(xmlFile)
            root = tree.getroot()
            
            filename = root[1].text
            width = int(root[4][0].text)
            height = int(root[4][1].text)

            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []
            classes_text = []
            classes = []
            

            for member in root.findall('object'):
                beer = member[0].text
                xmin = int(member[4][0].text)
                ymin = int(member[4][1].text)
                xmax = int(member[4][2].text)
                ymax = int(member[4][3].text)
                
                xmins.append(xmin/width)
                xmaxs.append(xmax/width)
                ymins.append(ymin/height)
                ymaxs.append(ymax/height)
                classes_text.append(beer.encode('utf8'))
                classes.append(class_text_to_int(beer))

            with tf.io.gfile.GFile(os.path.join(path_input, '{}'.format(filename)), 'rb') as fid:
                encoded_jpg = fid.read()
            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
                'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature(IMAGE_FORMAT),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
            }))
            
            writer.write(tf_example.SerializeToString())
    writer.close()             
    output_path = os.path.join(os.getcwd(), path_output)
    print('Successfully created the TFRecords: {}'.format(output_path))

xml_to_tf(PATH_TEST, PATH_RECORD_TEST)
xml_to_tf(PATH_TRAIN, PATH_RECORD_TRAIN)
