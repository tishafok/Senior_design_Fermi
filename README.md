**Training Inception SSD V2 Object Detection model with TensorFlow 1.14 on custom "nuts" and "bolts" 
images dataset.**

Dataset generated from CAD Rendering for Fermilab National Accelerator Laboratory autonomous maintenance project.
Trained on UIC Dragon supercomputer, deployed on Nvidia Jetson Nano with Intel RealSense D435 Stereoscopic camera

Achieved 0.857 mean Average Precision (mAP), 0.699 mean Average Recall (mAR)

Instructions:
- Install Python 3.6 and TensorFlow 1.14 (CPU version)
- Follow 
[this Guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md "This Guide") to install other librarier & download the tensorflow/models repository.
- Create folder "Dataset" with 2 sub folders: "Test" and "Train".
- Generate dataset images and divide them 80/20 into train/test.
- Use LabelImg to create XML label files for images.
- In **TFR_records.py** edit PATH_TEST and PATH_TRAIN to point to Test/Train folders. Set names for PATH_RECORD_TEST & PATH_RECORD_TRAIN
- To generate .records files run: *python3 TFR_records.py*
- Download [Inception V2 files](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz) 
- Set: *cd ~/tensorflow/models-master/research/object_detection*
- Create "Training" folder that contains:
  1. **pipeline.config** and **labelmap.pbtxt** files from [this repository/Final_Model](https://github.com/tishafok/Senior_design_Fermi/tree/master/Final_Model)
  3. **model.ckpt-xxxx** files (where xxxx is latest training step)
  4. **train.record** & **test.record**
 - Set: *cd to ~/research* then set slim directories path
 *export PYTHONPATH =$PYTHINPATH:`pwd`:`pwd`/slim*
 - Set: *cd to ~/object_detection*
 - To initialize training run: 
 
   *python3 model_main.py \
   --logtostderr \
   --model_dir=Training/ \
   --pipeline_config_path=Training/pipeline.config* 
 
- When training is complete use **export_inference_graph.py** to generate **frozen_inference_graph.pb**
- Run: 
   *python3 export_inference_graph.py \
   --input_type image_tensor \
   --pipeline_config_path Training/pipeline.config \
   --trained_checkpoint_prefix Training/model.ckpt-xxxx \
   --output_directory GRAPH_DIR*

- Edit PATH_TO_CKPT & PATH_TO_LABELS in **Fermi_Detection.py** to point to **frozen_inference_graph.pb** and **labelmap.pbtxt** files
- To begin inference run: *python3 Fermi_Detection.py* 


Other files: 

**Dragon_job_sub_file** (used to submit model training job to Dragon supercomputer via Moab/Torque job scheduler)

**List of libraries** (list of all libraries and version used to set dependencies for training and deployment)

**Model_image_tester.py** (static script to check model recall accuracy)

**inception_model_trained_logs** (generated during training to check model training progress and accuracy)

**tf_text_graph_ssd.py** (used to generate protobuf txt file for OpenCV DNN module deployment with Intel RealSense D435 camera)
