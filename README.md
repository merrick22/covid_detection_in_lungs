# Covid Detection In Lungs Using CNN

## CNN - Convoluted Neural Networks

## Procedure : 
### 1. Data Preparation and Importing Libraries : 
- **numpy, matplotlib.pyplot, os, keras, sklearn, seaborn, tensorflow**: These libraries are imported for various purposes such as data handling, visualization, file handling, deep learning model creation, metrics calculation, and saving/loading models.
- **Train_path, Test_path**: These variables store the paths to the training and test datasets.

### 2. Image Visualization : 
- **Image Visualization**: This section displays an example image from the dataset using matplotlib.

### 3. Image to Data Conversion :
- **ImageDataGenerator**: Instances of ImageDataGenerator are created for both the training and test datasets to convert RGB images into datasets and rescale them.

### 4. Data Visualization :
- **Org_train_dataset, Org_test_dataset**: ImageDataGenerator.flow_from_directory() is used to create dataset batches for training and test datasets. It also visualizes the different classes present in the dataset.

### 5. CNN Model Creation :
- **Sequential Model**: A sequential model is created using Keras Sequential API.
- **Conv2D, MaxPool2D, Dropout, Flatten, Dense**: These layers are added sequentially to the model to create a convolutional neural network.
- **model.compile**: The model is compiled with binary cross-entropy loss function, Adam optimizer, and accuracy metrics.

### 6. Model Training :
- **train_datagen, test_dataset**: ImageDataGenerators are created for data augmentation and rescaling.
- **train_generator, test_generator**: ImageDataGenerator.flow_from_directory() is used to generate batches of training and test data.
- **model.fit**: The model is trained using the fit() function with the training generator and validation data.

### 7. Model Evaluation :
- **model.evaluate**: The trained model is evaluated on the test data to calculate loss and accuracy.

### 8. Model Saving :
- **model.save**: The trained model is saved to a file using the save() function.

### 9. Graphical Representation of Output :
- **Accuracy and Loss Graphs**: Matplotlib is used to visualize the training and validation accuracy and loss over epochs.

### 10. Loading the Saved Model :
- **tf.keras.models.load_model**: The saved model is loaded from memory.

### 11. Prediction :
- **loaded_model.predict**: The loaded model is used to make predictions on new images.

### 12. Model Visualization of Test Dataset :
- **Predicted Classes and True Classes**: Predicted classes are compared with true classes to identify any mismatches.
- **metrics.classification_report**: Classification report is generated to evaluate model performance.
- **confusion_matrix**: Confusion matrix is generated to visualize classification results.

## Conclusion :
The provided code demonstrates the process of building a Convolutional Neural Network (CNN) for COVID-19 detection using image data. It includes data preparation, model creation, training, evaluation, and prediction. Additionally, it visualizes model performance using accuracy, loss graphs, and confusion matrix.
