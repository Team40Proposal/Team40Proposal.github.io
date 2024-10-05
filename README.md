# Optimizing Attribute Selection for 3D Model Popularity Prediction
## Project Report
### Introduction and Background
With the expansion of 3D digital content creation, many model repositories and markets are evolving. The need for data-driven approaches to optimize decision-making has become increasingly critical. The choice of tags like product name, price, etc. to maximize model popularity is crucial.\
<br>
Studies show the effectiveness of content-based methods in organizing and retrieving 3D models and leveraging metadata/semantic features to enhance classification [1][2]. We will utilize ML models to automate decision making on attribute associations. By analyzing metadata (products' existing description tags), our system will predict the optimal attributes for new models. This provides publishers with insights into the factors driving product popularity, facilitating better decision-making and enhancing content marketing strategies [3][4].

### Dataset Description
Our dataset provides well-organized, structured information about 3D models, like names, URLs, pricing, and user engagement metrics (comments and likes). This format allows for efficient retrieval and analysis of model recommendations, tailored for further use in ML model training workflows. This data was ethically obtained in compliance with 3dsky's robots.txt.\
<br>
<a href="https://drive.google.com/file/d/1jdKMc4G1djuLWZy0UvU5rw932eBVn5Q7/view?usp=drive"> Link to Dataset </a>\
<br>
<a href="https://3dsky.org/3dmodels"> Link to Data Source </a>

### Problem Definition and Motivation
With increasing implementation of 3D model repositories and virtual asset marketing, there is a growing need for efficient market prediction methods. Publishers need reliable methods to determine the popularity of newly published products. We will develop an architecture capable of predicting the optimal combination of attributes that maximizes model popularity. 

### Methods
#### Data Preprocessing Methods
1. Standardization: Standardization ensures each feature has a mean 0 and standard deviation 1. Standardization will pre-process our datasets to improve performance. It will ensure that some attributes like price (with wide range) do not dominate other attributes.
2. One-Hot Encoding: Categorical variables (product categories, brands, etc.) are converted into a binary matrix. No ordinal relationship is considered between categories. This method will pre-process our data for performance enhancement of our model.
3. Imputation: Imputation pre-processing techniques ensure completeness of data. The size of dataset can be optimally reduced, ensuring better performance.

#### Machine Learning Models
1. Convolutional Neural Networks (CNNs): CNNs are an important deep learning architecture for image recognition. This architecture is characterized by having subsampling and convolutional layers and the training process updates the kernel weights. Convolutional neural networks are effective at extracting local features in images.
2. Multi-Input Neural Networks: While the image data is fed to a CNN, the non-image data can be fed to another neural network. The outputs of both neural networks can then be fed to an activation layer for classification.
3. K-Means: This technique will group the attributes of products into clusters, based on their role in determining the popularity of the published products. This will help identify which attributes are more important for optimizing popularity. 

### Expected Results and Discussion
The model is expected to deliver high accuracy/precision and low RMSE which indicates the effectiveness of attribute selection in predicting product popularity. Multiple assessment metrics will be implemented.

1. Accuracy provides a generalized assessment by measuring correct predictions over total predictions.  
2. Precision minimizes false positives, ensuring that 3D models not likely to become popular (and specific attributes) are not recommended. Recall reduces false negatives to make sure popular models and attributes are not overlooked. An F1 score will be calculated as a harmonic mean to provide a balanced evaluation metric.
3. A confusion matrix will help analyze true/false positives/negatives, offering a look into the model’s strengths and areas of improvement.  
4. Root mean squared error assesses the difference between predicted and actual popularity scores to see how the model handles outputs. Lower RMSE indicates close predictions and higher RMSE indicates which inputs point to areas of improvement.  

In this project, our goal is to operate efficient models to minimize computational resources for sustainability. We also aim to operate ethically; for example, our data set has been ethically sourced.

### Gantt Chart
<a href="https://gtvault.sharepoint.com/:x:/s/MachineLearningCS7641/EaLiTgVlemVKnLifPSGEGbMBKy4zsQolP880C8xhN7b61g?e=wzoIbL"> Link to Gantt Chart </a>

### Contribution Table

| Name | Proposal Contributions |
|------|------------------------|
| Kamyar Fatemifar | Problem Definition, Methods, Results, GitHub Repository |
| Suchismitaa Chakraverty | Introduction/Background, Methods, Results, References |
| Joseph Hardin | Methods, Results, Contribution Table, Report Website, Video Presentation |
| Abhishek Misra | Methods, Results, Discussion |
| Max Pan | Methods, Results, Gantt Chart |

### References
[1] Y. Yang, H. Lin., and Y. Zhang, "Content-Based 3D Model Retrieval: A Survey," *IEEE Trans. Syst., Man, Cybern. C,* vol. 37, no. 6, pp. 1081-1098, Nov. 2007, doi: 10.1109/TSMCC.2007.905756\
<br>
[2] J. Flotynski and K. Walczak, "Customization of 3D content with semantic meta-scenes," *Graphical Models*, vol. 88, pp. 23-39, Nov. 2016, doi: 10.1016/j.gmod.2016.07.001\
<br>
[3] C.E. Catalano, M. Mortara, M. Spagnuolo, and B. Falcidieno, "Semantics and 3D media: Current issues and perspectives," *Computers & Graphics*, vol. 35, no. 4, pp. 867-877, Aug. 2011, doi: 10.1016/j.cag.2011.03.040\
<br>
[4] S. Biasotti, A. Cerri, A. Bronstein, and M. Bronstein, "Recent Trends, Applications, and Perspectives in 3D Shape Similarity Assessment," *Computer Graphics Forum*, vol. 35, no. 6, pp. 87-119, Sep. 2016, doi: 10.1111/cgf.12734\

## Video Presentation
<a href="google.com"> Link to Video Presentation </a>

## GitHub Repository
<a href="https://github.com/kamyar94/Team40_L7641_Fall2024"> Link to GitHub Repository </a>
