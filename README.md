# Optimizing Attribute Selection for 3D Model Popularity Prediction
## Project Report
### Introduction/Background and Literature Review
With the expansion of 3D digital content creation landscape, many established model repositories and markets are evolving. Thus, the need for data-driven approaches to optimize decision making has become increasingly critical. The choice of proper descriptive product tags like product name, price, and other factors that can maximize the popularity of a model has correspondingly become a crucial need.\
<br>
Studies show the effectiveness of content-based methods in organizing and retrieving 3D models, particularly when leveraging metadata and semantic features to enhance classification [1], [2]. In this project, we utilize popular ML models and techniques to automate the process of decision making on attributes associations. By analyzing metadata (the existing tags associated with products), the proposed system will be able to predict the optimal attributes for new models. This, in turn, will provide product publishers with insights into the factors driving model popularity, facilitating better decision-making and enhancing content marketing strategies [3], [4].

### Dataset Description
Our dataset provides well-organized, structured information about 3D models, like names, URLs, pricing, and user engagement metrics such as comments and likes. This structured format allows for efficient retrieval and analysis of model recommendations, tailored for further use in ML model training workflows. This data was ethically obtained in compliance with 3dsky's robots.txt.\
<br>
<a href="https://drive.google.com/file/d/1jdKMc4G1djuLWZy0UvU5rw932eBVn5Q7/view?usp=drive"> Link to Dataset </a>\
<br>
<a href="https://3dsky.org/3dmodels"> Link to Data Source </a>

### Problem Definition and Motivation
There are many businesses that conduct marketing of virtual products such as 3D models. Most of these businesses rely on creating a platform for connecting model publishers with customers. In this scenario, publishers and platform owners value higher sales - and one of the most important factors affecting sales is the descriptive product tags which help customers identify the best products based on their needs. For every product published, publishers need to determine an ideal set of tags such that their products’ popularity get maximized, which consequently increases the sales of their products and overall market performance. 

In this project, we are trying to determine the ideal set of tags/attributes that maximize the popularity score and recommendation score of a product that a publisher is about to publish in a particular model sub-category. Popularity score is a feature that has been engineered based on number of likes, number of recommendations (provided by users for each product), number of comments, the sentiment of each comment (positive, negative or doubtful), the number of tags used for a product, and the publication date of that product. Random forest has been used to determine the weights of the factors determining the popularity score. After generating a popularity score, a recommendation score is calculated based on how many times a specific product URL has been recommended by other models on the website (it is important to note that this engineered recommendation score is different from the above recommendation count mentioned). Using these 2 scores and K-means clustering, we can determine the ideal set of tags that contribute to higher recommendation and popularity score for each product in a specific sub-category. 

### Methods
#### Data Preprocessing Methods
We have used the following pre-processing methods to clean our datasets. For this report, we have 2 models implemented: Random Forest and K-means. Both models use different pre-processing methods, which are listed below. 

##### Random Forest
Data Cleaning: While working on the datasets, we found various models with empty tags, models without categories and models with missing data. As a result, we employed this technique to remove those models to make our dataset cleaner and more effective for training purposes. String slicing has been used to extract values required for the later implementation of popularity score generation, random forest and K-means.\
<br>
Standardization: One of our features for the K-means model is popularity score, which we standardize such that the value of the popularity score lies between 0 to 10. To accomplish this, we developed a Python script which gets the maximum score and minimum score in every sub-category and calculates the new weighted popularity score using the following equation: 


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
