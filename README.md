# Data Science Project Portfolio

This repository presents a collection of *my personal data science projects* in the form of iPython Notebooks. This portfolio does not include my (academic) research-related projects for the confidentiality reasons. While the topics and datasets being explored in this portfolio are chosen based on my own interest, tha primary focus of the projects is to employ various approaches and tools of data analysis/modelling to extract buried stories of the datasets on hand.

While each jupyter notebook is self-explanatory, the required modules can be found in the "requirement.txt" contained in each project folder. If you wish to install the requirements, simply run `pip install -r requirements.txt`. 

Any questions or feedback regarding this portfolio can be kindly directed to the author, Sean Choi, at ss.choi@mail.utoronto.ca. 

## End-to-end projects
#### *tools: keras, tensorflow, scikit-learn, Pandas, Matplotlib, Seaborn*

* **[Morphology-based skin cancer classification][1]**: Designed a customized CNN model and implemented a transfer learning on VGG16 that achieved ~ 80% accuracy in classifying 7 different skin cancer types. While the target variables were highly imbalanced, the final model constructed was shown to well differentiate different classes soley based on their morphology. Intermediate layers of CNN were also visualized for deeper understaning of the system. Best model/weights have been saved in the file.
* **[Forest Fire area prediction][2]**: Constructed regression model using XGBooster regressor that can estimate the burning area from the future forest fire. In order to deal with highly imbalanced target variable, oversampling approach was taken to help the model being sensitive to a small chance of forest fire occurance. With the corresponding feature selection process, the model could achieve the RMSE of ~3.2.
* **[Motion-sensor-based human motion prediction and subject identification][3]**: Analyzed motion sensor signals from human subjects in Time-domain and Frequency-domain to confirm their differentiability. Furthermore, various statistical technique like t-SNE was employed to visualize how different motions of different subjects fall into the same cluster. XGBooster classifier was trained to predict the motion and even identify the specific subject with >95% accuracy with only using the small portion of the available data. The most useful sensors in general in terms of predicting specific motions and subject were gyrometers on arms and ankles. 
* **[Malignant breat tumor detection][4]**: As an extension of my Master's project, malignant breast tumor detection problem was explored using KNN and SVM classifiers. The metrics was carefully chosen so that to optimize the model that can avoid as much false-positives (predict as beningn while it is malignant) as possible. The performance of the two constructed models were compared with a that of a dummy classifer. While the KNN achieved ~ 99% precision, a highly tuned SVM classifier could achieve 100% precision while compromising its recall to some extent. 
* **[Retail sales prediction][5]**: Constructed an ensemble model to predict a purchase amount of new potential customers based on their low-level information such as Gender and Age group. A risk of data leakage, along with a careful feature engineering/selection, was investigated.  


## Data analysis/visualization
#### *tools: scikit-learn, Pandas, Matplotlib, Seaborn*
* **[911 call type][11]**: The 911 call dataset was cleaned and organized by implementing various feature engineering/extraction techniques.
* **[stock price][12]**: Analyzed daily returns of FANG (Facebook, Amazon, Netflix and Google) stocks between 2013/01/01 and 2018/01/01. A brief EDA on the fetched data could reveal the information that can help in future investment.


## Mini capstone project <font size=3> - quick baseline model construction for fast prototyping</font> 
#### *tools: tensorflow, scikit-learn, Pandas, Matplotlib, Seaborn*
* **[Bank note authentication prediction][21]**: Roughly constructed ***DNN*** was employed to differentiate the authentic and fake bank notes. Its classification accuracy was compared to that of highly-tuned logistic regression model to test its performance.  
* **[yelp review star prediction][22]**: Implemented ***NLP*** technique for processing the raw text data from Yelp. Quickly tested a few techniques like TF-IDF transformation to investigate their effectiveness on predicting the number of stars from the reviews using multinomial ***Naive Bayes classifier***.
* **[College type prediction][23]**: Famous clustering algorithms such as ***K-means and Agglomerative clusterings*** have been implemented in order to separate two groups of colleages (public and private). With the true labels, the clustering performance of each model has been evaluated.
* **[Advertisement click prediction][24]**: Designed and tuned the simple ***logistic regression*** model in order to predict if the new user would click the advertisement. With a tuning of regularization parameter the prediction accuracy could improve to ~97% from ~89% with the minimum effort.
* **[Anonymous data classification][25]**: ***KNN*** model was trained in order to classify the given anonymous dataset. The hyperparameter has been optimized using a grid search.
* **[Loan payback prediction][26]**: Constructued and compared basic ***tree-based models*** for their performance on prediction of loan payback based on LendingClub profiles. 

[1]:https://github.com/sungsujaing/DataScience_Portfolio/blob/master/SkinCancerClassification_CNN/SkinCancerClassification.ipynb
[2]:https://github.com/sungsujaing/DataScience_Portfolio/blob/master/EstimatingDamageFromForestFire/Forest_Fire_Prediction_Model.ipynb
[3]:https://github.com/sungsujaing/DataScience_Portfolio/blob/master/HumanAndMotionPrediction/Mobile_Human_Motion_Prediction.ipynb
[4]:https://github.com/sungsujaing/DataScience_Portfolio/blob/master/PrognosticBreaseTumorDetection/BenignBreatTumorDetection.ipynb
[5]:https://github.com/sungsujaing/DataScience_Portfolio/blob/master/RetailSalesPrediction_BlackFridayAnalysis/BlackFriday%2BAnalysis_Prediction.ipynb

[11]:https://github.com/sungsujaing/DataScience_Portfolio/blob/master/Mini%20capstone%20projects/EDA-911call_Montgomery.ipynb
[12]:https://github.com/sungsujaing/DataScience_Portfolio/blob/master/Mini%20capstone%20projects/EDA-FANG_StockPrice.ipynb

[21]:https://github.com/sungsujaing/DataScience_Portfolio/blob/master/Mini%20capstone%20projects/Bank%20authentication%20prediction_DNN%20buildup%20on%20TensorFlow.ipynb
[22]:https://github.com/sungsujaing/DataScience_Portfolio/blob/master/Mini%20capstone%20projects/Yelp_Review%20classification_NLP.ipynb
[23]:https://github.com/sungsujaing/DataScience_Portfolio/blob/master/Mini%20capstone%20projects/College%20type%20prediction_K-Means%20and%20Agglomerative%20Clustering.ipynb
[24]:https://github.com/sungsujaing/DataScience_Portfolio/blob/master/Mini%20capstone%20projects/Ad%20click%20prediction_Logistic%20Regression.ipynb
[25]:https://github.com/sungsujaing/DataScience_Portfolio/blob/master/Mini%20capstone%20projects/AnonymousDataClassification_KNN.ipynb
[26]:https://github.com/sungsujaing/DataScience_Portfolio/blob/master/Mini%20capstone%20projects/Loan_payback_prediction_Decision%20Trees%20and%20Random%20Forest.ipynb
