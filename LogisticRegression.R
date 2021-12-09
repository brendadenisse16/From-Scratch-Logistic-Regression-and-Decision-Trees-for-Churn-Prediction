

mydata=read.csv("C:/Users/Jeanne/Desktop/Mast 679 - H/term project/churn.csv")

#convert a data.frame the data imported 
churn.df<- data.frame(mydata) 

#  We can remove "RowNumber","CustomerId","Surname" attributes, which are not appropriate for classification features
churn.df<-churn.df[,!names(churn.df)%in% c("RowNumber","CustomerId","Surname")]

# Creating Additional Predictors
attach(churn.df)
CreditScoreGivenAge <- CreditScore/Age

newratio<- data.frame(
  +  CreditScoreGivenAge)

#Adding new variable to churn data frame
churn.df <- cbind(churn.df, newratio)
head(churn.df)
attach(churn.df)

#Creating indicator variables for categorical variables
#install.packages("dummies")
library(dummies) 

churn.new <- dummy.data.frame(churn.df, sep = ".")
names(churn.new)

#However, we will omit one of the dummy variables for Geography and Gender when we use machine-learning technique, 
#to avoid multicollinearity and to recover non-singularity of our design. (to keep the independence of the variable)
#For Example if GeographySpain is 0 and GeographyFrance is 0 means that the customer is from Germany, because he/she neither Spain and France
#  We can remove "Geography.Germany","Gender.Male", 
churn.df<-churn.new[,!names(churn.df)%in% c("Geography.Germany","Gender.Male")]
churn.df<-churn.new[,!names(churn.df)%in% c("Geography.Germany","Gender.Male")]

# #Renaming columns to print the tree decision and it be more interpretable
label_rename <- rbind(
  c("CreditScore","X1"),
  c("Age","X2"),
  c("Tenure","X3"),
  c("Balance","X4"),
  c("NumOfProducts","X5"),
  c("EstimatedSalary","X6"),
  c("X.CreditScoreGivenAge","X7"),
  c("Geography.France","X8"),
  c("Geography.Spain","X9"),
  c("Gender.Female","X10"),
  c("HasCrCard","X11"),
  c("IsActiveMember","X12"),
  c("Exited","Y")
)
colnames(churn.df) <-
  sapply(colnames(churn.df), function(x) label_rename[label_rename[, 1]==x, 2])

#  Scaling the continuous variables
# function to scale the continuous variables
predictorscale<-function(x){return(scale(x, center = TRUE, scale = TRUE)) } 

continuous_variable = c(1,5,6,7,8,11,13) # index of the columns of continuous variables
churn.scaled.df = predictorscale(churn.df[,continuous_variable]) # dataset with continuous variables scaled
# If you want to see what is the mean and deviation standard used to scale the predictors :
center <- attr(churn.scaled.df,"scaled:center") # the variable means (means that were substracted)
scale <-attr(churn.scaled.df,"scaled:scale") # the variable standard deviations (the scaling applied to each variable )

scaling<-cbind(center,scale)
colnames(scaling)<- c("mean","std.dev")
scaling



categorical_variable =c(2,3,4,9,10,12) # index of the columns of categorical variables
churn.categorical.variables = churn.df[,categorical_variable] # dataset with categorical variables

###########################################################
##### FINAL DATA SET THAT WE WILL USE FOR MODEL FITTING##
##########################################################
#merge continuous variables scaled and categorical variables
churn.data <- cbind(churn.scaled.df,churn.categorical.variables)


###########################################################
################## LOGISTIC REGRESSION - FUNCTIONS#########
##########################################################
# Logistic Link Function
logisticf = function(x){return(1 / ( 1 + exp(-x) ))   } 

# Function to get the estimating the parameters  betas
negloglikelihoodLogistic = function(betavec,ResponseY,DesignX){
  
  zetavec= logisticf( DesignX %*% matrix(betavec,ncol=1) )
  logpdfobs = dbinom(x = ResponseY, size=1, prob = zetavec, log = TRUE) 
  negloglik = - sum(logpdfobs)
  
  return(negloglik)
}


#######################################################################
############### LOGISTIC REGRESSION TREE WITHOUT CROSS VALIDATION ###########
#######################################################################

###### FIT MODEL #######################
######PARAMETERS #######################
#Set random seed. Don't remove this line.
set.seed(195)

# Shuffle the dataset; build train and test
n <- nrow(churn.data)

# SPLIT DATA SET 80% TRAINING - 20% TESTING
shuffled <- churn.data[sample(n),]
train <- shuffled[1:round(0.8 * n),]
test <- shuffled[(round(0.8 * n) + 1):n,]

ntrain = dim(train)[1]
ntest=dim(test)[1] 

#####################################################
############ MODEL 1 ###############################
####################################################
# Model 1 : Considering all variables continuous 
#* The first model is built based on numeric variables: Credit Score,Tenure,Balance,Num Of Products, 
#Estimated Salary, Age and Credit Score given Age.

predictors = c(1,2,3,4,5,6,7) # index of the columns of continuous variables
# Beta's Estimated 
DesignXtrain1 = data.matrix(cbind(rep(1,ntrain),train[, predictors])) # this matrix contain all the predictor variables 
initbetas=rep(0,dim(DesignXtrain1)[2]) #initial value to start the optimization

Optimizenegloglik=nlm(negloglikelihoodLogistic,initbetas,ResponseY=train$Y, DesignX=DesignXtrain1 ) #optimize the negative log-likelihood
OptimBetas = Optimizenegloglik$estimate #optima betas
print(OptimBetas)

#Making predictions with Logistic Regression
#########CONFUSION MATRICES/ERROR RATES#################

#Parameters
n=dim(churn.data)[1]
ntrain = dim(train)[1]
ntest=dim(test)[1] #num  observations in the test set
nchurners = sum(test$Y)#total number of churners
nnochurners = ntest - nchurners #total number not in churn

#Matrix X Design (test data)
DesignXtest1 = data.matrix(cbind(rep(1,ntest),test[, predictors]))

# Get probabilities using logistic function and betas obtained in training set with own function
ProbChurnTest = logisticf(DesignXtest1 %*% matrix(OptimBetas,ncol=1) )

# Loss Function (loss of misclassifying a churn observation)
lossdefvec = c(1,20,50)

for (ll in lossdefvec){
  
  thresh = 1/(1+ll)
  
  flaggedobs = ProbChurnTest>=thresh
  
  TP= sum(test$Y[flaggedobs==TRUE]) #true positives
  FP= sum(1-test$Y[flaggedobs==TRUE])#false positives
  TN= sum(1-test$Y[flaggedobs==FALSE])#true negatives
  FN= sum(test$Y[flaggedobs==FALSE])#false negatives
  
  ConfusionMatrix = matrix(c(TN,FN,FP,TP),ncol=2,byrow=TRUE )
  print(ConfusionMatrix)
  
  ErrorRateLoss = (FP +FN)/ntest
  print(ErrorRateLoss)
  
}

###################################################
#ROC Curve (Test set)

OrderedChurner = test$Y[order(ProbChurnTest, decreasing =TRUE)] #resort the default vector by PD
TruePositiveRate = cumsum(OrderedChurner)/nchurners
FalsePositiveRate = cumsum(1-OrderedChurner)/nnochurners

plot( FalsePositiveRate, TruePositiveRate, 
      type='s', main='ROC Curve (Test set)',
      ylab='True positive rate',
      xlab='False positive rate')
lines(0:1,0:1,col='red',lty=2,lwd=2)

####AUROC

Churnindex = which(OrderedChurner==1)
nchurners = sum(test$Y)
nnochurners = length(test$Y) - nchurners

#####HERE, HORIZONTAL RECTANGLES ARE CONSIDERED (INSTEAD)
#####OF VERTICAL RECTANGLES AS IN THE CLASS NOTES
Cumnochurn = Churnindex - 1:(length(Churnindex)) #cumulative number of non-churner at each churner (obs ordered by PD)
AUROC = sum(nnochurners - Cumnochurn)/ (nchurners * nnochurners)
print(AUROC)


#####################################################
############ MODEL 2 ###############################
####################################################
# Model 2 : Considering all categorical variables:Geography, Gender, HasCrCard and IsActiveMember.

predictors = c(8,9,10,11,12) # index of the columns of categorical 
# Beta's Estimated 
DesignXtrain2 = data.matrix(cbind(rep(1,ntrain),train[, predictors])) # this matrix contain all the predictor variables 
initbetas=rep(0,dim(DesignXtrain2)[2]) #initial value to start the optimization

Optimizenegloglik=nlm(negloglikelihoodLogistic,initbetas,ResponseY=train$Y, DesignX=DesignXtrain2 ) #optimize the negative log-likelihood
OptimBetas = Optimizenegloglik$estimate #optima betas
print(OptimBetas)

#Making predictions with Logistic Regression
#########CONFUSION MATRICES/ERROR RATES#################

#Parameters
n=dim(churn.data)[1]
ntrain = dim(train)[1]
ntest=dim(test)[1] #num  observations in the test set
nchurners = sum(test$Y)#total number of churners
nnochurners = ntest - nchurners #total number not in churn

#Matrix X Design (test data)
DesignXtest2 = data.matrix(cbind(rep(1,ntest),test[, predictors]))

# Get probabilities using logistic function and betas obtained in training set with own function
ProbChurnTest = logisticf(DesignXtest2 %*% matrix(OptimBetas,ncol=1) )

# Loss Function (loss of misclassifying a churn observation)
lossdefvec = c(1,20,50)

for (ll in lossdefvec){
  
  thresh = 1/(1+ll)
  
  flaggedobs = ProbChurnTest>=thresh
  
  TP= sum(test$Y[flaggedobs==TRUE]) #true positives
  FP= sum(1-test$Y[flaggedobs==TRUE])#false positives
  TN= sum(1-test$Y[flaggedobs==FALSE])#true negatives
  FN= sum(test$Y[flaggedobs==FALSE])#false negatives
  
  ConfusionMatrix = matrix(c(TN,FN,FP,TP),ncol=2,byrow=TRUE )
  print(ConfusionMatrix)
  
  ErrorRateLoss = (FP +FN)/ntest
  print(ErrorRateLoss)
  
}
#ErrorRateLoss l=1 -->
#ErrorRateLoss l=20 --> 
#ErrorRateLoss l=50 --> 

###################################################
#ROC Curve (Test set)

OrderedChurner = test$Y[order(ProbChurnTest, decreasing =TRUE)] #resort the default vector by PD
TruePositiveRate = cumsum(OrderedChurner)/nchurners
FalsePositiveRate = cumsum(1-OrderedChurner)/nnochurners

plot( FalsePositiveRate, TruePositiveRate, 
      type='s', main='ROC Curve (Test set)',
      ylab='True positive rate',
      xlab='False positive rate')
lines(0:1,0:1,col='red',lty=2,lwd=2)

####AUROC

Churnindex = which(OrderedChurner==1)
nchurners = sum(test$Y)
nnochurners = length(test$Y) - nchurners

#####HERE, HORIZONTAL RECTANGLES ARE CONSIDERED (INSTEAD)
#####OF VERTICAL RECTANGLES AS IN THE CLASS NOTES
Cumnochurn = Churnindex - 1:(length(Churnindex)) #cumulative number of non-churner at each churner (obs ordered by PD)
AUROC = sum(nnochurners - Cumnochurn)/ (nchurners * nnochurners)
print(AUROC)


#####################################################
############ MODEL 3 ###############################
####################################################
#Model 3 is built, using the most significant variables associated asymptotic p-values from t-tests.
#Age, Tenure, EstimatedSalary and HasaCard don't significant predictors.

predictors = c(1,4,5,7,8,9,10,12) # index of the columns of categorical 
# Beta's Estimated 
DesignXtrain3 = data.matrix(cbind(rep(1,ntrain),train[, predictors])) # this matrix contain all the predictor variables 
initbetas=rep(0,dim(DesignXtrain3)[2]) #initial value to start the optimization

Optimizenegloglik=nlm(negloglikelihoodLogistic,initbetas,ResponseY=train$Y, DesignX=DesignXtrain3 ) #optimize the negative log-likelihood
OptimBetas = Optimizenegloglik$estimate #optima betas
print(OptimBetas)

#Making predictions with Logistic Regression
#########CONFUSION MATRICES/ERROR RATES#################

#Parameters
n=dim(churn.data)[1]
ntrain = dim(train)[1]
ntest=dim(test)[1] #num  observations in the test set
nchurners = sum(test$Y)#total number of churners
nnochurners = ntest - nchurners #total number not in churn

#Matrix X Design (test data)
DesignXtest3 = data.matrix(cbind(rep(1,ntest),test[, predictors]))

# Get probabilities using logistic function and betas obtained in training set with own function
ProbChurnTest = logisticf(DesignXtest3 %*% matrix(OptimBetas,ncol=1) )

# Loss Function (loss of misclassifying a churn observation)
lossdefvec = c(1,20,50)

for (ll in lossdefvec){
  
  thresh = 1/(1+ll)
  
  flaggedobs = ProbChurnTest>=thresh
  
  TP= sum(test$Y[flaggedobs==TRUE]) #true positives
  FP= sum(1-test$Y[flaggedobs==TRUE])#false positives
  TN= sum(1-test$Y[flaggedobs==FALSE])#true negatives
  FN= sum(test$Y[flaggedobs==FALSE])#false negatives
  
  ConfusionMatrix = matrix(c(TN,FN,FP,TP),ncol=2,byrow=TRUE )
  print(ConfusionMatrix)
  
  ErrorRateLoss = (FP +FN)/ntest
  print(ErrorRateLoss)
  
}
#ErrorRateLoss l=1 
#ErrorRateLoss l=20 
#ErrorRateLoss l=50 

###################################################
#ROC Curve (Test set)

OrderedChurner = test$Y[order(ProbChurnTest, decreasing =TRUE)] #resort the default vector by PD
TruePositiveRate = cumsum(OrderedChurner)/nchurners
FalsePositiveRate = cumsum(1-OrderedChurner)/nnochurners

plot( FalsePositiveRate, TruePositiveRate, 
      type='s', main='ROC Curve (Test set)',
      ylab='True positive rate',
      xlab='False positive rate')
lines(0:1,0:1,col='red',lty=2,lwd=2)

####AUROC

Churnindex = which(OrderedChurner==1)
nchurners = sum(test$Y)
nnochurners = length(test$Y) - nchurners

#####HERE, HORIZONTAL RECTANGLES ARE CONSIDERED (INSTEAD)
#####OF VERTICAL RECTANGLES AS IN THE CLASS NOTES
Cumnochurn = Churnindex - 1:(length(Churnindex)) #cumulative number of non-churner at each churner (obs ordered by PD)
AUROC = sum(nnochurners - Cumnochurn)/ (nchurners * nnochurners)
print(AUROC)


#####################################################
############ MODEL 4 ###############################
####################################################
#Model 4 only takes best six variables, according to the BIC criterion. 
#It only takes Age, Balance, Geography, Gender, IsActiveMember.

predictors = c(2,4,8,9,10,12) # index of the columns of categorical 
# Beta's Estimated 
DesignXtrain4 = data.matrix(cbind(rep(1,ntrain),train[, predictors])) # this matrix contain all the predictor variables 
initbetas=rep(0,dim(DesignXtrain4)[2]) #initial value to start the optimization

Optimizenegloglik=nlm(negloglikelihoodLogistic,initbetas,ResponseY=train$Y, DesignX=DesignXtrain4 ) #optimize the negative log-likelihood
OptimBetas = Optimizenegloglik$estimate #optima betas
print(OptimBetas)

#Making predictions with Logistic Regression
#########CONFUSION MATRICES/ERROR RATES#################

#Parameters
n=dim(churn.data)[1]
ntrain = dim(train)[1]
ntest=dim(test)[1] #num  observations in the test set
nchurners = sum(test$Y)#total number of churners
nnochurners = ntest - nchurners #total number not in churn

#Matrix X Design (test data)
DesignXtest4= data.matrix(cbind(rep(1,ntest),test[, predictors]))

# Get probabilities using logistic function and betas obtained in training set with own function
ProbChurnTest = logisticf(DesignXtest4 %*% matrix(OptimBetas,ncol=1) )

# Loss Function (loss of misclassifying a churn observation)
lossdefvec = c(1,20,50)

for (ll in lossdefvec){
  
  thresh = 1/(1+ll)
  
  flaggedobs = ProbChurnTest>=thresh
  
  TP= sum(test$Y[flaggedobs==TRUE]) #true positives
  FP= sum(1-test$Y[flaggedobs==TRUE])#false positives
  TN= sum(1-test$Y[flaggedobs==FALSE])#true negatives
  FN= sum(test$Y[flaggedobs==FALSE])#false negatives
  
  ConfusionMatrix = matrix(c(TN,FN,FP,TP),ncol=2,byrow=TRUE )
  print(ConfusionMatrix)
  
  ErrorRateLoss = (FP +FN)/ntest
  print(ErrorRateLoss)
  
}
#ErrorRateLoss l=1 --> 
#ErrorRateLoss l=20 -->  
#ErrorRateLoss l=50 -->  

###################################################
#ROC Curve (Test set)

OrderedChurner = test$Y[order(ProbChurnTest, decreasing =TRUE)] #resort the default vector by PD
TruePositiveRate = cumsum(OrderedChurner)/nchurners
FalsePositiveRate = cumsum(1-OrderedChurner)/nnochurners

plot( FalsePositiveRate, TruePositiveRate, 
      type='s', main='ROC Curve (Test set)',
      ylab='True positive rate',
      xlab='False positive rate')
lines(0:1,0:1,col='red',lty=2,lwd=2)

####AUROC

Churnindex = which(OrderedChurner==1)
nchurners = sum(test$Y)
nnochurners = length(test$Y) - nchurners

#####HERE, HORIZONTAL RECTANGLES ARE CONSIDERED (INSTEAD)
#####OF VERTICAL RECTANGLES AS IN THE CLASS NOTES
Cumnochurn = Churnindex - 1:(length(Churnindex)) #cumulative number of non-churner at each churner (obs ordered by PD)
AUROC = sum(nnochurners - Cumnochurn)/ (nchurners * nnochurners)
print(AUROC)




#####################################################
############ MODEL 5 ###############################
####################################################
#Model 5 is built based on all variables (numeric and categorical).

predictors = c(1,2,3,4,5,6,7,8,9,10,11,12) # index of the columns of categorical 
# Beta's Estimated 
DesignXtrain5 = data.matrix(cbind(rep(1,ntrain),train[, predictors])) # this matrix contain all the predictor variables 
initbetas=rep(0,dim(DesignXtrain5)[2]) #initial value to start the optimization

Optimizenegloglik=nlm(negloglikelihoodLogistic,initbetas,ResponseY=train$Y, DesignX=DesignXtrain5 ) #optimize the negative log-likelihood
OptimBetas = Optimizenegloglik$estimate #optima betas
print(OptimBetas)

#Making predictions with Logistic Regression
#########CONFUSION MATRICES/ERROR RATES#################

#Parameters
n=dim(churn.data)[1]
ntrain = dim(train)[1]
ntest=dim(test)[1] #num  observations in the test set
nchurners = sum(test$Y)#total number of churners
nnochurners = ntest - nchurners #total number not in churn

#Matrix X Design (test data)
DesignXtest5= data.matrix(cbind(rep(1,ntest),test[, predictors]))

# Get probabilities using logistic function and betas obtained in training set with own function
ProbChurnTest = logisticf(DesignXtest5 %*% matrix(OptimBetas,ncol=1) )

# Loss Function (loss of misclassifying a churn observation)
lossdefvec = c(1,20,50)

for (ll in lossdefvec){
  
  thresh = 1/(1+ll)
  
  flaggedobs = ProbChurnTest>=thresh
  
  TP= sum(test$Y[flaggedobs==TRUE]) #true positives
  FP= sum(1-test$Y[flaggedobs==TRUE])#false positives
  TN= sum(1-test$Y[flaggedobs==FALSE])#true negatives
  FN= sum(test$Y[flaggedobs==FALSE])#false negatives
  
  ConfusionMatrix = matrix(c(TN,FN,FP,TP),ncol=2,byrow=TRUE )
  print(ConfusionMatrix)
  
  ErrorRateLoss = (FP +FN)/ntest
  print(ErrorRateLoss)
  
}
#ErrorRateLoss l=1 --> 
#ErrorRateLoss l=20 -->  
#ErrorRateLoss l=50 --> 

###################################################
#ROC Curve (Test set)

OrderedChurner = test$Y[order(ProbChurnTest, decreasing =TRUE)] #resort the default vector by PD
TruePositiveRate = cumsum(OrderedChurner)/nchurners
FalsePositiveRate = cumsum(1-OrderedChurner)/nnochurners

plot( FalsePositiveRate, TruePositiveRate, 
      type='s', main='ROC Curve (Test set)',
      ylab='True positive rate',
      xlab='False positive rate')
lines(0:1,0:1,col='red',lty=2,lwd=2)

####AUROC

Churnindex = which(OrderedChurner==1)
nchurners = sum(test$Y)
nnochurners = length(test$Y) - nchurners

#####HERE, HORIZONTAL RECTANGLES ARE CONSIDERED (INSTEAD)
#####OF VERTICAL RECTANGLES AS IN THE CLASS NOTES
Cumnochurn = Churnindex - 1:(length(Churnindex)) #cumulative number of non-churner at each churner (obs ordered by PD)
AUROC = sum(nnochurners - Cumnochurn)/ (nchurners * nnochurners)
print(AUROC)


#######################################################################
############### LOGISTIC REGRESSION TREE WITH CROSS-VALIDATION ########
#######################################################################

### SEQUENTIAL VALIDATION

#Set random seed. Don't remove this line.
set.seed(195)
# Shuffle the dataset; build train and test
n <- nrow(churn.data)
shuffled <- churn.data[sample(n),]

# Number of fold in CV
kfold=10

#####################################################
############ MODEL 1 ###############################
####################################################
# Model 1 : Considering all variables continuous 
#* The first model is built based on numeric variables: Credit Score,Tenure,Balance,Num Of Products, 
#Estimated Salary, Age and Credit Score given Age.


predictors = c(1,2,3,4,5,6,7) # index of the columns of continuous variables
AUROC<-rep(0,kfold)
ErrorRateLoss<-rep(0,kfold)  #Type-1 Error rate
#Accuracy<-rep(0,kfold)     #Type-2 Error rate
ConfusionMatrixFold = matrix(0,nrow=4,ncol=kfold )
row.names(ConfusionMatrixFold)<-c("TN","FN","FP","TP")

# dividing  data in 10 times (10 different train and test data) and --
#then running regression logistic model on train model and predicting churn on test data--
# then checking aveg error loss rate and avg AUROC

for (i in 1:kfold) {
  
  # These indices indicate the interval of the test set
  i=1
  indices <- (((i-1) * round((1/kfold)*nrow(shuffled))) + 1):((i*round((1/kfold) * nrow(shuffled))))
  
  # Exclude them from the train set
  train <- shuffled[-indices,]
  # Include them in the test set
  test <- shuffled[indices,]
  
  #Predict by Logistic Regression
  ntrain = dim(train)[1]
  ntest=dim(test)[1] 
  
  DesignXtrain1 = data.matrix(cbind(rep(1,ntrain),train[, predictors]))
  initbetas=rep(0,dim(DesignXtrain1)[2]) #initial value to start the optimization
  Optimizenegloglik=nlm(negloglikelihoodLogistic,initbetas,ResponseY=train$Y, DesignX=DesignXtrain1 )  #optimize the negative log-likelihood
  OptimBetas = Optimizenegloglik$estimate #optimal betas
  
  nchurners = sum(test$Y) #total number of churners
  nnochurners = length(test$Y) - nchurners  #total number not churners
  
  #Matrix X Design (test data)
  ntrain = dim(train)[1]
  ntest=dim(test)[1] 
  DesignXtest1 = data.matrix(cbind(rep(1,ntest),test[, predictors]))
  
  
  # Get probabilities using logistic function and betas obtained in training set with own function
  ProbChurnTest = logisticf(DesignXtest1 %*% matrix(OptimBetas,ncol=1) )
  
  
  # Loss Function (loss of misclassifying a churn observation)
  l=50
  thresh = 1/(1+l)
  flaggedobs = ProbChurnTest>=thresh
  TP= sum(test$Y[flaggedobs==TRUE]) #true positives
  FP= sum(1-test$Y[flaggedobs==TRUE])#false positives
  TN= sum(1-test$Y[flaggedobs==FALSE])#true negatives
  FN= sum(test$Y[flaggedobs==FALSE])#false negatives
  ConfusionMatrixFold[,i]<-t(cbind(FN,TN,FP,TP))
  ErrorRateLoss[i] = (FP +FN)/ntest # loss of misclassifying a churn observation
  
  
  #####AUROC(Test set)
  OrderedChurner = test$Y[order(ProbChurnTest, decreasing =TRUE)] #resort the default vector by PChurner
  Churnindex = which(OrderedChurner==1)
  
  Cumnochurn = Churnindex - 1:(length(Churnindex)) #cumulative number of non-churner at each churner (obs ordered by PD)
  AUROC[i] = sum(nnochurners - Cumnochurn)/ (nchurners * nnochurners)
  
}

################################################  
## Summary Performance of Regression Logistic ##
# Mean of Error Type 1, 2 and AUROC
AvgAUROC<-mean(AUROC)
AvgAUROC #0.7375721

# To get the index of Kfold where we got the maximum AUROC
maxAUROC = which.max(AUROC)
AUROC[maxAUROC] #0.7612818

AvgErrorRateLoss<-mean(ErrorRateLoss)
AvgErrorRateLoss #0.7687


#####################################################
############ MODEL 2 ###############################
####################################################

# Model 2 : Considering all categorical variables:Geography, Gender, HasCrCard and IsActiveMember.

predictors = c(8,9,10,11,12) # index of the columns of categorical 
AUROC<-rep(0,kfold)
ErrorRateLoss<-rep(0,kfold)  #Type-1 Error rate
#Accuracy<-rep(0,kfold)     #Type-2 Error rate
ConfusionMatrixFold = matrix(0,nrow=4,ncol=kfold )
row.names(ConfusionMatrixFold)<-c("TN","FN","FP","TP")


for (i in 1:kfold) {
  
  # These indices indicate the interval of the test set
  # i=10
  indices <- (((i-1) * round((1/kfold)*nrow(shuffled))) + 1):((i*round((1/kfold) * nrow(shuffled))))
  
  # Exclude them from the train set
  train <- shuffled[-indices,]
  # Include them in the test set
  test <- shuffled[indices,]
  
  #Predict by Logistic Regression
  ntrain = dim(train)[1]
  ntest=dim(test)[1] 
  
  DesignXtrain2 = data.matrix(cbind(rep(1,ntrain),train[, predictors]))
  initbetas=rep(0,dim(DesignXtrain2)[2]) #initial value to start the optimization
  Optimizenegloglik=nlm(negloglikelihoodLogistic,initbetas,ResponseY=train$Y, DesignX=DesignXtrain2)  #optimize the negative log-likelihood
  OptimBetas = Optimizenegloglik$estimate #optimal betas
  
  nchurners = sum(test$Y) #total number of churners
  nnochurners = length(test$Y) - nchurners  #total number not churners
  
  #Matrix X Design (test data)
  ntrain = dim(train)[1]
  ntest=dim(test)[1] 
  DesignXtest2 = data.matrix(cbind(rep(1,ntest),test[, predictors]))
  
  
  # Get probabilities using logistic function and betas obtained in training set with own function
  ProbChurnTest = logisticf(DesignXtest2 %*% matrix(OptimBetas,ncol=1) )
  
  
  # Loss Function (loss of misclassifying a churn observation)
  l=50
  thresh = 1/(1+l)
  flaggedobs = ProbChurnTest>=thresh
  TP= sum(test$Y[flaggedobs==TRUE]) #true positives
  FP= sum(1-test$Y[flaggedobs==TRUE])#false positives
  TN= sum(1-test$Y[flaggedobs==FALSE])#true negatives
  FN= sum(test$Y[flaggedobs==FALSE])#false negatives
  ConfusionMatrixFold[,i]<-t(cbind(FN,TN,FP,TP))
  ErrorRateLoss[i] = (FP +FN)/ntest # loss of misclassifying a churn observation
  #Accuracy[i] = (TP +TN)/(TP+FP+TN+FN) # # Accuracy (all correct / all) 
  # 
  
  #####AUROC(Test set)
  OrderedChurner = test$Y[order(ProbChurnTest, decreasing =TRUE)] #resort the default vector by PChurner
  Churnindex = which(OrderedChurner==1)
  
  Cumnochurn = Churnindex - 1:(length(Churnindex)) #cumulative number of non-churner at each churner (obs ordered by PD)
  AUROC[i] = sum(nnochurners - Cumnochurn)/ (nchurners * nnochurners)
  
}


################################################  
## Summary Performance of Regression Logistic ##
# Mean of Error Type 1, 2 and AUROC
AvgAUROC<-mean(AUROC)
AvgAUROC #0.6727255

# To get the index of Kfold where we got the maximum AUROC
maxAUROC = which.max(AUROC)
AUROC[maxAUROC] #0.6953025

AvgErrorRateLoss<-mean(ErrorRateLoss)
AvgErrorRateLoss #0.7963


#####################################################
############ MODEL 3 ###############################
####################################################
#Model 3 is built, using the most significant variables associated asymptotic p-values from t-tests.
#Age, Tenure, EstimatedSalary and HasaCard don't significant predictors.


predictors = c(1,4,5,7,8,9,10,12) # index of the columns of categorical 
AUROC<-rep(0,kfold)
ErrorRateLoss<-rep(0,kfold)  #Type-1 Error rate
#Accuracy<-rep(0,kfold)     #Type-2 Error rate
ConfusionMatrixFold = matrix(0,nrow=4,ncol=kfold )
row.names(ConfusionMatrixFold)<-c("TN","FN","FP","TP")


for (i in 1:kfold) {
  
  # These indices indicate the interval of the test set
  # i=10
  indices <- (((i-1) * round((1/kfold)*nrow(shuffled))) + 1):((i*round((1/kfold) * nrow(shuffled))))
  
  # Exclude them from the train set
  train <- shuffled[-indices,]
  # Include them in the test set
  test <- shuffled[indices,]
  
  #Predict by Logistic Regression
  ntrain = dim(train)[1]
  ntest=dim(test)[1] 
  
  DesignXtrain3 = data.matrix(cbind(rep(1,ntrain),train[, predictors]))
  initbetas=rep(0,dim(DesignXtrain3)[2]) #initial value to start the optimization
  Optimizenegloglik=nlm(negloglikelihoodLogistic,initbetas,ResponseY=train$Y, DesignX=DesignXtrain3)  #optimize the negative log-likelihood
  OptimBetas = Optimizenegloglik$estimate #optimal betas
  
  nchurners = sum(test$Y) #total number of churners
  nnochurners = length(test$Y) - nchurners  #total number not churners
  
  #Matrix X Design (test data)
  ntrain = dim(train)[1]
  ntest=dim(test)[1] 
  DesignXtest3 = data.matrix(cbind(rep(1,ntest),test[, predictors]))
  
  
  # Get probabilities using logistic function and betas obtained in training set with own function
  ProbChurnTest = logisticf(DesignXtest3 %*% matrix(OptimBetas,ncol=1) )
  
  
  # Loss Function (loss of misclassifying a churn observation)
  l=50
  thresh = 1/(1+l)
  flaggedobs = ProbChurnTest>=thresh
  TP= sum(test$Y[flaggedobs==TRUE]) #true positives
  FP= sum(1-test$Y[flaggedobs==TRUE])#false positives
  TN= sum(1-test$Y[flaggedobs==FALSE])#true negatives
  FN= sum(test$Y[flaggedobs==FALSE])#false negatives
  ConfusionMatrixFold[,i]<-t(cbind(FN,TN,FP,TP))
  ErrorRateLoss[i] = (FP +FN)/ntest # loss of misclassifying a churn observation
  #Accuracy[i] = (TP +TN)/(TP+FP+TN+FN) # # Accuracy (all correct / all) 
  # 
  
  #####AUROC(Test set)
  OrderedChurner = test$Y[order(ProbChurnTest, decreasing =TRUE)] #resort the default vector by PChurner
  Churnindex = which(OrderedChurner==1)
  
  Cumnochurn = Churnindex - 1:(length(Churnindex)) #cumulative number of non-churner at each churner (obs ordered by PD)
  AUROC[i] = sum(nnochurners - Cumnochurn)/ (nchurners * nnochurners)
  
}


################################################  
## Summary Performance of Regression Logistic ##
# Mean of Error Type 1, 2 and AUROC
AvgAUROC<-mean(AUROC)
AvgAUROC # 0.7723905

# To get the index of Kfold where we got the maximum AUROC
maxAUROC = which.max(AUROC)
AUROC[maxAUROC] # 0.7915172

AvgErrorRateLoss<-mean(ErrorRateLoss)
AvgErrorRateLoss #0.7537


#####################################################
############ MODEL 4###############################
####################################################
#Model 4 only takes best six variables, according to the BIC criterion. 
#It only takes Age, Balance, Geography, Gender, IsActiveMember.


predictors = c(2,4,8,9,10,12)  # index of the columns of categorical 
AUROC<-rep(0,kfold)
ErrorRateLoss<-rep(0,kfold)  #Type-1 Error rate
#Accuracy<-rep(0,kfold)     #Type-2 Error rate
ConfusionMatrixFold = matrix(0,nrow=4,ncol=kfold )
row.names(ConfusionMatrixFold)<-c("TN","FN","FP","TP")


for (i in 1:kfold) {
  
  # These indices indicate the interval of the test set
  # i=10
  indices <- (((i-1) * round((1/kfold)*nrow(shuffled))) + 1):((i*round((1/kfold) * nrow(shuffled))))
  
  # Exclude them from the train set
  train <- shuffled[-indices,]
  # Include them in the test set
  test <- shuffled[indices,]
  
  #Predict by Logistic Regression
  ntrain = dim(train)[1]
  ntest=dim(test)[1] 
  
  DesignXtrain4 = data.matrix(cbind(rep(1,ntrain),train[, predictors]))
  initbetas=rep(0,dim(DesignXtrain4)[2]) #initial value to start the optimization
  Optimizenegloglik=nlm(negloglikelihoodLogistic,initbetas,ResponseY=train$Y, DesignX=DesignXtrain4)  #optimize the negative log-likelihood
  OptimBetas = Optimizenegloglik$estimate #optimal betas
  
  nchurners = sum(test$Y) #total number of churners
  nnochurners = length(test$Y) - nchurners  #total number not churners
  
  #Matrix X Design (test data)
  ntrain = dim(train)[1]
  ntest=dim(test)[1] 
  DesignXtest4 = data.matrix(cbind(rep(1,ntest),test[, predictors]))
  
  
  # Get probabilities using logistic function and betas obtained in training set with own function
  ProbChurnTest = logisticf(DesignXtest4 %*% matrix(OptimBetas,ncol=1) )
  
  
  # Loss Function (loss of misclassifying a churn observation)
  l=50
  thresh = 1/(1+l)
  flaggedobs = ProbChurnTest>=thresh
  TP= sum(test$Y[flaggedobs==TRUE]) #true positives
  FP= sum(1-test$Y[flaggedobs==TRUE])#false positives
  TN= sum(1-test$Y[flaggedobs==FALSE])#true negatives
  FN= sum(test$Y[flaggedobs==FALSE])#false negatives
  ConfusionMatrixFold[,i]<-t(cbind(FN,TN,FP,TP))
  ErrorRateLoss[i] = (FP +FN)/ntest # loss of misclassifying a churn observation
  #Accuracy[i] = (TP +TN)/(TP+FP+TN+FN) # # Accuracy (all correct / all) 
  # 
  
  #####AUROC(Test set)
  OrderedChurner = test$Y[order(ProbChurnTest, decreasing =TRUE)] #resort the default vector by PChurner
  Churnindex = which(OrderedChurner==1)
  
  Cumnochurn = Churnindex - 1:(length(Churnindex)) #cumulative number of non-churner at each churner (obs ordered by PD)
  AUROC[i] = sum(nnochurners - Cumnochurn)/ (nchurners * nnochurners)
  
}


################################################  
## Summary Performance of Regression Logistic ##
# Mean of Error Type 1, 2 and AUROC
AvgAUROC<-mean(AUROC)
AvgAUROC # 0.7659751

# To get the index of Kfold where we got the maximum AUROC
maxAUROC = which.max(AUROC)
AUROC[maxAUROC] # 0.792208
> 
  
  AvgErrorRateLoss<-mean(ErrorRateLoss)
AvgErrorRateLoss # 0.7918


#####################################################
############ MODEL 5###############################
####################################################
#Model 5 is built based on all variables (numeric and categorical).


predictors = c(1,2,3,4,5,6,7,8,9,10,11,12)  # index of the columns of categorical 
AUROC<-rep(0,kfold)
ErrorRateLoss<-rep(0,kfold)  #Type-1 Error rate
#Accuracy<-rep(0,kfold)     #Type-2 Error rate
ConfusionMatrixFold = matrix(0,nrow=4,ncol=kfold )
row.names(ConfusionMatrixFold)<-c("TN","FN","FP","TP")

for (i in 1:kfold) {
  
  # These indices indicate the interval of the test set
  # i=10
  indices <- (((i-1) * round((1/kfold)*nrow(shuffled))) + 1):((i*round((1/kfold) * nrow(shuffled))))
  
  # Exclude them from the train set
  train <- shuffled[-indices,]
  # Include them in the test set
  test <- shuffled[indices,]
  
  #Predict by Logistic Regression
  ntrain = dim(train)[1]
  ntest=dim(test)[1] 
  
  DesignXtrain5 = data.matrix(cbind(rep(1,ntrain),train[, predictors]))
  initbetas=rep(0,dim(DesignXtrain5)[2]) #initial value to start the optimization
  Optimizenegloglik=nlm(negloglikelihoodLogistic,initbetas,ResponseY=train$Y, DesignX=DesignXtrain5)  #optimize the negative log-likelihood
  OptimBetas = Optimizenegloglik$estimate #optimal betas
  
  nchurners = sum(test$Y) #total number of churners
  nnochurners = length(test$Y) - nchurners  #total number not churners
  
  #Matrix X Design (test data)
  ntrain = dim(train)[1]
  ntest=dim(test)[1] 
  DesignXtest5 = data.matrix(cbind(rep(1,ntest),test[, predictors]))
  
  
  # Get probabilities using logistic function and betas obtained in training set with own function
  ProbChurnTest = logisticf(DesignXtest5 %*% matrix(OptimBetas,ncol=1) )
  
  
  # Loss Function (loss of misclassifying a churn observation)
  l=50
  thresh = 1/(1+l)
  flaggedobs = ProbChurnTest>=thresh
  TP= sum(test$Y[flaggedobs==TRUE]) #true positives
  FP= sum(1-test$Y[flaggedobs==TRUE])#false positives
  TN= sum(1-test$Y[flaggedobs==FALSE])#true negatives
  FN= sum(test$Y[flaggedobs==FALSE])#false negatives
  ConfusionMatrixFold[,i]<-t(cbind(FN,TN,FP,TP))
  ErrorRateLoss[i] = (FP +FN)/ntest # loss of misclassifying a churn observation
  #Accuracy[i] = (TP +TN)/(TP+FP+TN+FN) # # Accuracy (all correct / all) 
  # 
  
  #####AUROC(Test set)
  OrderedChurner = test$Y[order(ProbChurnTest, decreasing =TRUE)] #resort the default vector by PChurner
  Churnindex = which(OrderedChurner==1)
  
  Cumnochurn = Churnindex - 1:(length(Churnindex)) #cumulative number of non-churner at each churner (obs ordered by PD)
  AUROC[i] = sum(nnochurners - Cumnochurn)/ (nchurners * nnochurners)
  
}

################################################  
## Summary Performance of Regression Logistic ##
# Mean of Error Type 1, 2 and AUROC
AvgAUROC<-mean(AUROC)
AvgAUROC # 0.7719649

# To get the index of Kfold where we got the maximum AUROC
maxAUROC = which.max(AUROC)
AUROC[maxAUROC] # 0.7904601
> 
  
  AvgErrorRateLoss<-mean(ErrorRateLoss)
AvgErrorRateLoss # 0.7541


