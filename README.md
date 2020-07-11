# Deep Neural Networks

Collection of machine learning projects. 

* [Image classification](#image-classification) using Keras and TensorFlow
* [Fake news detection](#fake-news-detection) using scikit-learn 
* [Colour detection](#colour-detection) using OpenCV
* [Neural networks](#neural-networks) using high-level TensorFlow implementation and low-level NumPy implementation

## Image Classification

Image classification using a deep convolutional neural networks (CNN). The code is at [src/CNN](src/CNN), and the main file is [deepNetwork.py](src/CNN/deepNetwork.py).

### Data Augmentation

![augmented](images/augmented.png)

### Model Summary

![summary](images/summary.png)

### Model Visualization

![model](images/model.png)

### Model Performance

![performance](images/performance.png)

## Fake News Detection

Trains classifier from fake news data and reports testing accuracy. The code is at [src/fakeNews](src/fakeNews), and the main file is [fakeNews.py](src/fakeNews/fakeNews.py).

### Prerequisites

The following Python libraries are required

* scikit-learn
* pandas

### Running

* Navigate to the directory with ```cd src``` and ```cd fakeNews```

* Run ```python fakeNews.py``` to train a classifier on fake news data. It will report the confusion matrix, accuracy, precision, recall, and f1 score.

![image](images/image.png)

## Colour Detection

Displays colours name and RGB values of clicked pixels. The code is at [src/colourDetection](src/colourDetection), and the main file is [colourDetection.py](src/colourDetection/colourDetection.py).

### Prerequisites

The following Python libraries are required

* OpenCV
* pandas

### Running

* Navigate to the directory with ```cd src``` and ```cd colourDetection```

* Run ```python colourDetection.py ``` to open up an image. Double click to display the RGB values and name of closest colour. Press esc to exit.

![image](images/image1.png)

## Neural Networks

* Deep neural network with [TensorFlow implementation](src/tfNetwork). Main file is [deepNetwork.py](src/tfNetwork/deepNetwork.py).
* Deep neural network with [NumPy implementation](src/npNetwork). Main file is [deepNetwork.py](src/npNetwork/deepNetwork.py).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
