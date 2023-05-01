# Mango-Leaf-Disease-Image-Classification
About the project: In this project, we have a dataset that consists of 3200 train images and 800 test images of mango leaves and our goal is to identify different the disease such as 'Anthracnose',  'Bacterial Canker',  'Cutting Weevil',  'Die Back',  'Gall Midge',  'Healthy',  'Powdery Mildew' and 'Sooty Mould' from the image. For this purpose we make use of CNN model that is build using fully connected layers, convolutional layers and pooling layers, Decision Tree, Multinomial Naive Bayes, K-Nearest Neighbors and Voting Classifier. CNN, Multinomial Naive Bayes, K-Nearest Neighbors, and Voting Classifier are all machine learning algorithms used for classification tasks. Each algorithm has its own strengths and weaknesses, and the choice of algorithm depends on the specific problem and data at hand.
1. Convolutional Neural Network (CNN):
CNN is a deep learning algorithm commonly used for image classification tasks. It consists of multiple layers of convolutions and pooling, followed by fully connected layers. CNNs can automatically learn relevant features from raw data, making them highly effective for image classification tasks.
2. Multinomial Naive Bayes:
Multinomial Naive Bayes is a probabilistic algorithm used for text classification tasks. It assumes that the features are independent of each other and follow a multinomial distribution. It is often used for text classification tasks, such as sentiment analysis, spam detection, and document classification.
3. K-Nearest Neighbors (KNN):
KNN is a non-parametric algorithm used for classification tasks. It classifies a new data point based on the class of its k nearest neighbors in the training data. The choice of k determines the size of the neighborhood used to classify the new data point. KNN is simple and easy to understand, but can be computationally expensive for large datasets.
4. Decision tree: 
It is a popular machine learning algorithm used for classification and regression tasks. It is a tree-like structure where each internal node represents a feature or attribute, each branch represents a possible value of that attribute, and each leaf node represents a class label or a numeric value. The algorithm works by recursively partitioning the data into smaller subsets based on the most informative feature or attribute. At each step, the algorithm selects the feature that best separates the data into the different classes or groups. The best feature is determined based on some measure of purity or impurity, such as entropy or Gini index.
6. Voting Classifier:
A Voting Classifier is an ensemble learning algorithm that combines multiple classification models to improve the overall performance. It takes the predictions from multiple models and outputs the class with the highest probability. It can be used with any combination of classification models, including CNN, Multinomial Naive Bayes, and KNN.

Installation Instructions: This does not require any specifi installations as the project is built over colab. However there are some libraries that need to be included for this project. Most of them are the basic libraries such as 'numpy' and 'pandas' which are a must include in every Machine Learning project.

How to run the project:

Open Google Colab in your web browser and sign in using your Google account.
Pull the code from the repository to your colab or you can directly click on 'https://colab.research.google.com/drive/1SF0T2tBdgCj2Vi8_6_neCFyq0j_pV3qo?usp=sharing' link for code. This is a view only notebook and you will have to create a copy of it in your local drive for editing.
Select a runtime type by clicking on "Runtime" in the top menu, then change to "GPU" for running faster.
Download the dataset from 'https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset/code' and upload it the drive.
Access the dataset from the drive by mounting it.
Rest of the instructions include running every code snippet and get the output.

Additional Information: This deals with a bulky dataset and stores huge amount of feature information for comparison. Thus it is recommended that one uses high RAM and faster runtime hardware for computation.
