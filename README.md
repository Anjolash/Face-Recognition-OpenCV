# Missing Person Search System

This project is aimed at developing a Missing Person Search System using Facial Recognition Technology. The system consists of two main components: training the facial recognition model and performing real-time facial recognition for identification purposes.

## Training the Model

The **train10.py** script is responsible for training the facial recognition model. It uses the OpenCV library to detect faces in images and preprocess them for training. The script follows the following steps:

1. Load the dataset: The script reads the images from the specified dataset directory and extracts the faces using the Viola-Jones face detection algorithm.

2. Preprocess the images: The extracted faces are resized to a uniform size for consistency during training. The images are stored in an array along with their corresponding labels.

3. Encode the labels: The labels are encoded using the LabelEncoder from the scikit-learn library to convert them into numerical values.

4. Split the dataset: The dataset is split into training and testing sets using the train_test_split function from scikit-learn.

5. Normalize the pixel values: The pixel values of the images are normalized to improve training performance.

6. Create the CNN model: The Convolutional Neural Network (CNN) model is constructed using the Sequential model from the Keras library. It consists of convolutional and pooling layers followed by fully connected layers.

7. Compile and train the model: The model is compiled with appropriate loss function, optimizer, and metrics, and then trained using the training dataset.

8. Save the model: The trained model is saved for future use.

## Real-time Facial Recognition

The **face2.0.py** script implements the real-time facial recognition functionality. It utilizes the trained model to recognize faces in a live video feed. The script performs the following steps:

1. Load the trained model: The pre-trained facial recognition model is loaded using the load_model function from Keras.

2. Load the Haar cascade classifiers: The Haar cascade classifiers for face, eye, and smile detection are loaded using the CascadeClassifier class from OpenCV.

3. Capture and process the video feed: The script captures frames from the video feed, converts them to grayscale, and detects faces using the face cascade classifier.

4. Perform facial recognition: For each detected face, the script resizes and normalizes the face image, passes it through the trained model, and predicts the class probabilities. The predicted class is mapped back to the original label.

5. Display the results: The script draws bounding boxes around the detected faces and displays the predicted class name and probability on the frame.

6. Print the predictions: After displaying the results, the script prints the predicted class and probability for each detected face.

## Requirements

To run this project, the following dependencies need to be installed:

- OpenCV
- NumPy
- scikit-learn
- Keras

Please refer to the respective documentation of each library for installation instructions.

## Usage

1. Prepare the dataset: Create a dataset directory containing subdirectories for each person's images. Place the respective images inside each subdirectory.

2. Run the training script: Execute the **train10.py** script to train the facial recognition model on the provided dataset. This will save the trained model as "facial_recognition_model3.h5".

3. Run the real-time recognition script: Execute the **face2.0.py** script to perform real-time facial recognition using the trained model. Ensure that the necessary Haar cascade classifier XML files are present in the specified locations.

4. View the results: The script will display the live video feed with bounding boxes around detected faces and their predicted class labels and probabilities.

Note: Make sure to adjust the paths and parameters in the scripts as per your system and requirements.

## Contributors

- Lasekan Anjolaoluwa Dominion(https://github.com/Anjolash)

Feel free to contribute to this project by creating pull requests or reporting issues.

## License

This project is licensed under the [MIT License](LICENSE).