Inception V2 Object Detection model training with TensorFlow file content:

- Dragon Job Sub File used to submit model training job to Dragon supercomputer via Moab/Torque job scheduler
- Fermi Detection script used to deploy trained model to detect the target objects and locate them in 3D coordinate space
- List of libraries is a list of all libraries and version used to set dependencies for training and deployment
- Model Image tester static script to check model recall accuracy
- TFR records scipt used to generate training and testing TF record files from images + corresponding XML labels
- Export inference graph copy (located in TF folder) used to freeze the trained model for deployment
- Inception Model trained logs are generated during training to check model training progress and accuracy
- TF text graph SSD used to generate protobuf txt file for OpenCV DNN module deployment (with Intel RealSense D435 camera)
