# Optimizing Attribute Selection for 3D Model Popularity Prediction
## Project Report
### Introduction and Background
With the expansion of 3D digital content creation, many model repositories and markets are evolving, so the need for data-driven approaches to optimize decision making has become increasingly critical. The choice of tags like product name, price, etc. to maximize model popularity is therefore crucial.\
<br>
Studies have shown the effectiveness of content-based methods in organizing and retrieving 3D models, particularly when leveraging metadata and semantic features to enhance classification [1], [2]. We will utilize popular ML models and techniques to automate decision making on attributes associations. By analyzing metadata (products' existing description tags), the system will be able to predict the optimal attributes for new models. This will provide product publishers insights into the factors driving model popularity, facilitating better decision-making and enhancing content marketing strategies [3], [4]​.

### Dataset Description
Our dataset provides a well-organized structure capturing information about 3D models, such as names, URLs, pricing, and user engagement metrics like comments and likes. This format allows for efficient retrieval and analysis of model recommendations, tailored for further use in ML model training workflows.\
<br>
<a href="https://drive.google.com/file/d/1jdKMc4G1djuLWZy0UvU5rw932eBVn5Q7/view?usp=drive"> Link to Dataset </a>\
<a href="https://3dsky.org/3dmodels"> Link to data source </a>

### Problem Definition and Motivation
With the increasing implementations of 3D model repositories and virtual asset marketing, there is a growing need for efficient market prediction methods. Product publishers need a reliable method to determine the popularity of newly published products. We aim to develop a ML model capable of predicting the optimal combination of attributes that maximizes model popularity based on available data. 

### Methods
#### Data Preprocessing Methods
1. Standardization: Standardization transforms each feature to have a mean of 0 and standard deviation of 1. We will use this technique to pre-process our datasets such that our model will perform better. Furthermore, it will ensure that some attributes like product price (with wide range) do not dominate other attributes like product name or product ratings.
2. One-Hot Encoding: Categorical variables (product categories, brands, etc.) are converted into a binary matrix. No ordinal relationship is considered between categories. We will use this method to pre-process our data for performance enhancement of our model.
3. Imputation: Imputation pre-processing technique will ensure completeness of data. The size of dataset can be optimally reduced, thus ensuring better performance.

#### Machine Learning Models
1. Convolutional Neural Networks (CNNs): CNNs are one of the primary deep learning architectures for image recognition. In addition, this architecture is characterized by having subsampling and convolutional layers and the training process updates the kernel weights. It is noted that convolutional neural networks are effective at extracting local features in images.
2. Multi-Input Neural Networks: While the image data is fed to a CNN, the non-image data can be fed to another neural network. The outputs of both neural networks can then be fed to an activation layer for classification.
3. K-Means: We will be using this technique to group the attributes of products into clusters, based on their role in determining the popularity of the products to be published in an online portal. This will help identify which attributes/tags are more important for maximizing product popularity. 

### Expected Results and Discussion
The model is expected to deliver high accuracy, precision, and a low RMSE, which together will indicate the effectiveness of attribute selection in predicting 3D model popularity. Multiple quantitative metrics will be implemented to assess the model’s performance.  

1. Accuracy will provide a generalized assessment by measuring correct predictions over total predictions.  
2. Precision will aim to minimize the false positives ensuring that 3D models not likely to become popular (and specific attributes) are not recommended. Alongside precision, recall will aim to reduce false negatives to make sure popular models and attributes are not overlooked. Then, an F1 score will be calculated as a harmonic mean to provide a balanced evaluation metric.  
3. A confusion matrix will aid in analyzing true positives, true negatives, false positives, and false negatives, offering a detailed look into where the model’s strengths (in prediction) are and what areas need improvement.  
4. Root mean squared error will assess the difference between predicted and actual popularity scores to see how the model handles outputs. A lower RMSE will indicate close predictions while a higher RMSE will indicate which inputs would point to areas of improvement for the model.  

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
<br>
[5] H. Laga, M. Mortara, and M. Spagnuolo, "Geometry and context for semantic correspondences and functionality recognition in man-made 3D shapes," *ACM Trans. Graph.*, vol. 32, no. 5, pp. 1-16, Sep. 2013, doi: 10.1145/2516971.2516975\
<br>
[6] S. Hijazi, R. Kumar, C. Rowen, "Using Convolutional Neural Networks for Image Recognition", *Computer Science*, 2015\
<br>
[7] A. R. Mesa, J. Y. Chiang, "Multi-Input Deep Learning Model with RGB and Hyperspectral Imaging for Banana Grading", *Agriculture*, July 2021.\
<br>
[8] L. Linhui, J. Weinpeng, W. Huihui, "Extracting the Forest Type From Remote Sensing Images by Random Forest", *IEEE*, Dec. 2020, 10.1109/JSEN.2020.3045501\

## Video Presentation
<a href="google.com"> Link to Video Presentation </a>

## GitHub Repository
<a href="https://github.com/kamyar94/Team40_L7641_Fall2024"> Link to GitHub Repository </a>
