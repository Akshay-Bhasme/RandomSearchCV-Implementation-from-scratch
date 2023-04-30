# RandomSearchCV-Implementation-from-scratch
This repository contains implementation of RandomSearchCV from scratch with 3 fold cross validation with KNN as classifier.

Cross Validation: Machine Learning
Understanding need of cross validation and the steps in implementing it from scratch
Building a well fitted machine learning model that is not overfitting as well as underfitting is one of the main challenges every data scientist faces. Just like salt is one of the ingredients that enhance the taste of curry and over addition of it ruins the taste, hyperparameters in any machine learning model decides whether model will be overfit, good fit or underfit. With the correct hyperparameter estimation one can easily develop generalized model that gives high performance on training as well as future datapoints passed to the model. This article explains the need selecting best parameters, need of cross validation and the step wise approach to implement it from scratch.
What does hyper-parameter mean?
The parameters that control the learning process of a machine learning model are know as hyperparameters. Hyperparameters are model specific that needs to be tuned to train a reliable model that will give high accuracy in predicting future query datapoints.
Let’s understand the concept of Underfit and Overfit from K-Nearest Neighbours example:
##### Underfit: 
Consider the scenario depicted below,
  
  ![image](https://user-images.githubusercontent.com/87875987/235347225-bc5ceebf-a8ee-4d8b-bdf0-bd8c20b369ca.png)

	                
Suppose k= 25 and data is imbalanced. In this case, model will always classify a Test (query) point as point from majority class, because in the k-neighbours there will be more points from majority class than minority class. By having larger ‘k’ value the test point will get misclassified and performance of the model on future data will be worst. So underfitting is the phenomenon where model becomes biased towards majority class as model learning do not happens and the model becomes black box, so ends up predicting wrongly.
From the accuracy perspective Overfit happens when train error and test error is high.

##### Overfit:

![image](https://user-images.githubusercontent.com/87875987/235347236-590b7ba1-103d-4120-b283-62f49132adb3.png)

Suppose k= 1. In this case, due to one outlier the test point gets misclassified. In the Overfit scenario model tries to fit every point even if it is outlier. That’s why when ‘k’ is small the probability of point being misclassified increases in the presence of outlier. So Overfit is the condition where model learning happens with respect to every point even if it is outlier which in future causes datapoints to be misclassified.
From the accuracy perspective Overfit happens when train error is low and test error is high. It predicts all the points from training data correctly while it fails to predict test data points.

##### Well Fit: 
![image](https://user-images.githubusercontent.com/87875987/235347242-0941ea3f-7c97-4f11-84eb-0bb5aac57ca9.png)

                                          
Suppose k= 6. In this case, the model learns well with the optimum ‘k’ neighbours which are required for the correct classification of the data points. In the image you can see even though there is one outlier near to the query data point (green point), due to optimum ‘k’ value data point will be classified correctly. Well fit is the scenario where model tries to learn properly that do not over react due to presence of outliers and classifies the datapoints correctly.
From the accuracy perspective, for well fitted model train error as well as test error will be low, and model shows high accuracy in predicting both train and future data points.

## Need for cross validation:
We need to select optimum ‘k’ parameter value to generalize the model which works well on future unseen data. We generally split the data in Train and Test, we train the model on training data and we test it on test data, but what is the guarantee that it will give same accuracy for classifying the future data? Suppose we use test data to select optimum ‘k’ value of the model and we get 96% accuracy on test data, but we cannot say that accuracy on future data is 96% as test data is being used to determine ‘k’ value not as future data to check the accuracy on unseen points. This do not guarantee for future data to give accuracy of 96%.
To solve this problem, we use a technique called as Cross Validation. In cross validation we divide main dataset as: 

![image](https://user-images.githubusercontent.com/87875987/235347324-263484e1-85d8-4d7a-9788-4b1e34b50afe.png)

                   
Using D_train we calculate nearest neighbours, using D_cv we decide best ‘k’ value and using D_test we calculate the accuracy. We use D_test as unseen data instead of using it to decide best ‘k’ value which help us predict the model performance on future data. The accuracy on this unseen data is called as generalization accuracy and the error is called as generalization error.

## K-Fold cross validation: 
  (K here is not the ‘k’ from K-NN algorithm)

Suppose data is splitted as:
D_train = 60% , D_cv = 20% , D_test = 20%. 
Here we are using only 60% of data (D_train) to train the model and 20% (D_cv) to select the best ‘k’ value for K-NN model. If we can use 80% of the data to train the model, model will have more datapoints for learning. So we use D_train and D_cv for training the model as well as finding the best ‘k’ parameter value.
We do it as: 
Take whole D_train and D_cv as :  D_train = D_train + D_cv , keep the D_test to use it as future unseen data.
Divide the data into K-folds, where K is the number of parts we divide our data into. Consider 4-fold cross validation here,
Divide the data (D_Train) into 4 equal parts as: 

![image](https://user-images.githubusercontent.com/87875987/235347378-7ac913d9-3dbe-41e9-9d2d-bef24300843a.png)


Now we want to find the best value of ‘k’ parameter (neighbours), but we don’t have D_cv for it. 
So here we will use the 3 parts of data as training data and 4th part we will take it as D_cv, in the next iteration we will alter the data and so on. So there will be 4 such iterations  for each value of ’k’ neighbours parameter. For each iteration, for each cv data we calculate accuracy.

![image](https://user-images.githubusercontent.com/87875987/235347401-b9fbfe35-d0fe-477a-bfe5-0dfc435f043e.png)


For each ‘k’ value we calculate average accuracy as:
For k=1, avg acc= (A1+A2+A3+A4)/4
For k=2, avg acc= (A1+A2+A3+A4)/4
……. And so on.
We plot the ‘k’ value Vs average accuracy and we decide best ‘k’ value which is giving max accuracy on both train and test data. 
So by using 80% of the data like above we are not wasting 20% for D_cv seperately to calculate ‘k’ value. Then we use D_test as unseen data and we calculate accuracy on that data.
There are two approaches for cross validation you can refer links mentioned below to know about them:
1) GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
2) RandomSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html  



