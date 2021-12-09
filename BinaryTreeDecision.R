
mydata=read.csv("churn.csv")


#libraries
library(dummies) #to convert categorical variables to indicator variables
library(pROC) # to get the AUROC
library(igraph) # to plot the binary decision tree



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


################################################
########## DECISION TREE - CHURN PREDICTION ###
################################################

# function calculates the Gini index.
gini_index <- function(y) {
  if (length(y) == 0) return(0)
  p <- table(y) / length(y)
  sum(p * (1 - p))
}

# We create function "gini_index_aggr" 
gini_index_aggr <- function(y, condition_threshold, func = gini_index) {
  n1 <- sum(condition_threshold) # sums the TRUE from the condition_threshold
  n2 <- length(condition_threshold) - n1 # represents the FALSE from the condition_threshold
  if (n1 == 0 & n2 == 0) {
    return(0)
  }
  n1 / (n1 + n2) * func(y[condition_threshold]) +
    n2 / (n1 + n2) * func(y[!condition_threshold])
}


# We create function "min_gini_index_split"  gets the min gain index aggregate for a single predictor.
min_gini_index_split <- function(y, x, func = gini_index) {
  best_change <- NA
  split_value <- NA
  is_numeric <- !(is.factor(x) | is.logical(x) | is.character(x))
  for (val in sort(unique(x))) {
    mask <- x == val
    if (is_numeric) mask <- x < val
    change <- gini_index_aggr(y, mask, func)
    # cat("val: ", val, " change: ", change, "\n")
    if (is.na(best_change) | change < best_change) {
      best_change <- change
      split_value <- val
    }
  }
  return(list("best_change" = best_change,"split_value" = split_value,"is_numeric" = is_numeric))
}

# We create function "best_predictor_split" returns a list with the information of the predictor variable and value with the min Gini index split.
best_predictor_split <- function(X, y) {
  results <- sapply(X, function(x) min_gini_index_split(y, x)) #we will apply 'sapplying' the function to get the information of the predictor variable and value with the min gini index split
  best_name <- names(which.min(results["best_change", ])) # to get the index of the name of the variable (predictor with min gini)
  best_result <- results[, best_name] # to get the value of the of the variable predictor with min gini
  best_result[["name"]] <- best_name # to get the name of the of the variable predictor with min gini
  best_result
}

# returns a logical vector (TRUE or FALSE) based on the best split information obtained on the observations dataset. 
#The TRUE values represents the observations considered for the left branch, FALSE to the right one.

get_best_mask <- function(X, best_predictor_list) {
  best_mask <- X[, best_predictor_list$name] == best_predictor_list$split_value
  if (best_predictor_list$is_numeric) {
    best_mask <- X[, best_predictor_list$name] < best_predictor_list$split_value
  }
  return(best_mask)
}

de_escal_val <- function(feat_name, escaled_value) {
  # val*std.dev + mean
  val <- escaled_value
  if(sum(rownames(scaling)==feat_name)>0)
    val <- escaled_value * scaling[feat_name, 2] + scaling[feat_name, 1]
  
  round(val,2)
}
#de_escal_val("X8", -0.989290388)

# This function provides a decision rule description for each tree level.
get_split_node_description <-
  function(is_left, is_leaf, split, predict_value) {
    
    is_numeric <- is.numeric(split$split_value)
    split_sign <- ifelse(!is_numeric, "=", ifelse(is_left, "<", ">="))
    desc_vector <- NULL
    if (is.null(split)) {
      desc_vector <- "root"
    } else {
      desc_val <- de_escal_val(split$name,split$split_value)
      desc_vector <-
        c(split$name, split_sign, desc_val)
    }
    if (is_leaf) {
      desc_vector <- c(desc_vector,"::", predict_value
      )
    }
    paste(desc_vector, collapse = " ")
}

# This function builds the decision tree in a recursive binary splitting and greedy approach.
built_tree <- function(X, y, current_depth = 0, is_left = F, last_split = NULL) {
  local_depth <- current_depth + 1
  split <- best_predictor_split(X, y)
  mask <- get_best_mask(X, split)
  is_leaf <- T
  left_branch <- NULL
  right_branch <- NULL
  predict_value <- NULL
  
  if (local_depth < max_depth && sum(mask) >= min_leaf_size && length(mask) - sum(mask) >= min_leaf_size) {
    is_leaf <- F
    left_branch <- built_tree(X[mask, ], y[mask], local_depth, T, split)
    right_branch <- built_tree(X[!mask, ], y[!mask], local_depth, F, split)
  }
  if (is_leaf) {
    #the prediction is the most prevalent class (to get this, we calculate what is the proportion of observations)
    predict_value <- names(which.max(table(y)))
  }
  description <- get_split_node_description(
    is_left, is_leaf, last_split, predict_value
  )
  list(
    "depth" = local_depth,
    "split" = last_split,
    "mask" = mask,
    "is_leaf" = is_leaf,
    "is_left" = is_left,
    "predict_value" = predict_value,
    "left" = left_branch,
    "right" = right_branch,
    "description" = description
  )
}

# This functiom prints the decision rules description.
print_node <- function(node, target) {
  tabs <- paste(rep("\t", node$depth - 1))
  cat(tabs, node$description, "\n")
  if (!is.null(node$left)) print_node(node$left)
  if (!is.null(node$right)) print_node(node$right)
}


### TESTING PHASE
# This function predicts the class for one row using the decision rules.
predict_dt_row <- function(row, node) {
  # if the root has branches, we start at the left one
  if(!node$is_leaf & !is.null(node$left)) {
    split_feature <- node$left$split$name
    split_value <- node$left$split$split_value
    if(row[split_feature] < split_value) {
      return(predict_dt_row(row,node$left))
    } else {
      return(predict_dt_row(row, node$right))
    }
  } 
  node$predict_value
}


# retrieves the predicted class for all the rows in the test data.
predict_dt <- function(features, tree) {
  apply(features, 1, function(row) predict_dt_row(row, tree))
}

# Identify the prediction type : FP, TN, TP, FN
pred_type_row <- function(pair) {
  real <- pair[1]
  pred <- pair[2]
  desc <- NULL
  if(real == 0) {
    if(pred == 1) desc <- "FP"
    else desc <- "TN"
  } else {
    if(pred == 1) desc <- "TP"
    else desc <- "FN"
  }
}

# all the functions above are needed to plot the decision tree
edges <- c()
add_tree_edge <- function(a, b) {
  edges <<- c(edges, c(a,b))
}
edges_description<- NULL
add_node_descriptions <- function(pos, desc) {
  edges_description <<- rbind(edges_description, cbind(pos,desc))
}

edge_id <- 0
get_edge_id <- function() {
  edge_id <<- edge_id + 1
  edge_id
}

get_tree_edges <- function(node, curr_edge_id) {
  if ( is.null(node$split)) {
    add_node_descriptions(curr_edge_id, node$description) #root
  }
  if ( !is.null(node$left) ) {
    left_index <- get_edge_id()
    add_tree_edge(curr_edge_id, left_index)
    add_node_descriptions(left_index, node$left$description)
    get_tree_edges(node$left, left_index)
    
  }
  if ( !is.null(node$right) ) {
    right_index <- get_edge_id()
    add_tree_edge(curr_edge_id, right_index)
    add_node_descriptions(right_index, node$right$description)
    get_tree_edges(node$right, right_index)
  }
}

plot_tree <- function(dtree) {
  edge_id <<- 0
  edges_description <<- c()
  edges <<- c()
  get_tree_edges(dtree, get_edge_id())
  labels <- 
    apply(label_rename, 1,function(x) paste(c(x[2],x[1]), collapse = ": "))
  
  g <- graph.empty (nrow(edges_description), directed = F) #creating empty plot
  g<-add.edges(g, edges) #add edges
  V(g)$name<-edges_description[,2]
  par(mar = c(0,0,0,0), ps=14,cex=1 )
  V(g)$color="white"
  plot(
    g, 
    layout = layout.reingold.tilford(g, root = 1, flip.y = T, circular = F),
    vertex.size=sapply(edges_description[,2], function(x) (nchar(x) * 2.1)),
    vertex.label.cex=c(0.6),
    vertex.shape="rectangle")
  legend("left", legend=labels, cex=0.5)
}

#######################################################################
############### BINARY DECISION TREE WITHOUT CROSS-VALIDATION ###########
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

#####################################################
############ MODEL 1 ###############################
####################################################
# Model 1 : Considering all continuous variables 
#* The first model is built based on numeric variables: Credit Score,Tenure,Balance,Num Of Products, 
#Estimated Salary, Age and Credit Score given Age.

predictors = c(1,2,3,4,5,6,7) # index of the columns of continuous variables 
min_leaf_size <- 5 # stopping criteria : Stop when no subregion contains more than five observations
max_depth <- 4 # 6, 10 max depth  of regression tree

tree <- built_tree(train[, predictors], train[, 13])
# Print Rules Decision
print_node(tree)

# Print Plot Tree
plot_tree(tree)

# To get the prediction on test set given decision rules
testdata <-cbind(test[ ,predictors],test[, 13])
prediction <- predict_dt(testdata[,1:(ncol(testdata)-1)],tree)

# To get the accuracy of the prediction (percentage) ChurnerPredicted/ChurnerReal
res <- cbind(test[, 13], as.numeric(prediction))
length(which(res[,1] == res[,2])) / nrow(testdata)

# Type Error 1 - Type Error 2
pred_ident <- table(apply(res, 1, pred_type_row))
true_pos_rate <- pred_ident[4] / (pred_ident[4] + pred_ident[1])
false_pos_rate <- pred_ident[2] / (pred_ident[3] + pred_ident[2])
true_pos_rate
false_pos_rate

#ROC CURVE
plot.roc(test[,13], as.numeric(prediction), legacy.axes = T, percent = T)
# AUROC
roc_obj<- roc(test[,13], as.numeric(prediction))
AUROC = auc(roc_obj)
AUROC


###########################################################
############ MODEL 2 ###############################
####################################################
# Model 2 : Considering all categorical variables:Geography, Gender, HasCrCard and IsActiveMember.

predictors = c(8,9,10,11,12) # index of the columns of categorical variables
min_leaf_size <- 5 # stopping criteria : Stop when no subregion contains more than five observations
max_depth <- 4 # 3,4, 6, 10,... etc max depth  of decision tree

tree <- built_tree(train[, predictors], train[, 13])
# Print Rules Decision
print_node(tree)

# Print Plot Tree
plot_tree(tree)

# To get the prediction on test set given decision rules
testdata <-cbind(test[ ,predictors],test[, 13])
prediction <- predict_dt(testdata[,1:(ncol(testdata)-1)],tree)

# To get the accuracy of the prediction (percentage) ChurnerPredicted/ChurnerReal
res <- cbind(test[, 13], as.numeric(prediction))
length(which(res[,1] == res[,2])) / nrow(testdata)

# Type Error 1 - Type Error 2
pred_ident <- table(apply(res, 1, pred_type_row))
true_pos_rate <- pred_ident[4] / (pred_ident[4] + pred_ident[1])
false_pos_rate <- pred_ident[2] / (pred_ident[3] + pred_ident[2])
true_pos_rate
false_pos_rate

#ROC CURVE
plot.roc(test[,13], as.numeric(prediction), legacy.axes = T, percent = T)
# AUROC
roc_obj<- roc(test[,13], as.numeric(prediction))
AUROC = auc(roc_obj)
AUROC


#####################################################
############ MODEL 3 ###############################
####################################################
#Model 3 is built, using the most significant variables associated asymptotic p-values from t-tests.
#Age, Tenure, EstimatedSalary and HasaCard don't significant predictors.

predictors = c(1,4,5,7,8,9,10,12) # index of the columns of predictors
min_leaf_size <- 5 # stopping criteria : Stop when no subregion contains more than five observations
max_depth <- 4 # 3,4, 6, 10,... etc max depth  of decision tree

tree <- built_tree(train[, predictors], train[, 13])

# Print Rules Decision
print_node(tree)
# Print Plot Tree
plot_tree(tree)
# To get the prediction on test set given decision rules
testdata <-cbind(test[ ,predictors],test[, 13])
prediction <- predict_dt(testdata[,1:(ncol(testdata)-1)],tree)

# To get the accuracy of the prediction (percentage) ChurnerPredicted/ChurnerReal
res <- cbind(test[, 13], as.numeric(prediction))
length(which(res[,1] == res[,2])) / nrow(testdata)

# Type Error 1 - Type Error 2
pred_ident <- table(apply(res, 1, pred_type_row))
true_pos_rate <- pred_ident[4] / (pred_ident[4] + pred_ident[1])
false_pos_rate <- pred_ident[2] / (pred_ident[3] + pred_ident[2])
true_pos_rate
false_pos_rate



#ROC CURVE
plot.roc(test[,13], as.numeric(prediction), legacy.axes = T, percent = T)
# AUROC
roc_obj<- roc(test[,13], as.numeric(prediction))
AUROC = auc(roc_obj)
AUROC


#####################################################
############ MODEL 4 ###############################
####################################################
#Model 3 is built, using the most significant variables associated asymptotic p-values from t-tests.
#Age, Tenure, EstimatedSalary and HasaCard don't significant predictors.

predictors = c(2,4,8,9,10,12)  # index of the columns of categorical 
min_leaf_size <- 5 # stopping criteria : Stop when no subregion contains more than five observations
max_depth <- 4# 3,4,6, 10,... etc max depth  of decision tree

tree <- built_tree(train[, predictors], train[, 13])
# Print Rules Decision
print_node(tree)

# Plot Tree
plot_tree(tree)


# To get the prediction on test set given decision rules
testdata <-cbind(test[ ,predictors],test[, 13])
prediction <- predict_dt(testdata[,1:(ncol(testdata)-1)],tree)

# To get the accuracy of the prediction (percentage) ChurnerPredicted/ChurnerReal
res <- cbind(test[, 13], as.numeric(prediction))
length(which(res[,1] == res[,2])) / nrow(testdata)

# Type Error 1 - Type Error 2
pred_ident <- table(apply(res, 1, pred_type_row))
true_pos_rate <- pred_ident[4] / (pred_ident[4] + pred_ident[1])
false_pos_rate <- pred_ident[2] / (pred_ident[3] + pred_ident[2])
true_pos_rate
false_pos_rate

#ROC CURVE
plot.roc(test[,13], as.numeric(prediction), legacy.axes = T, percent = T)
# AUROC
roc_obj<- roc(test[,13], as.numeric(prediction))
AUROC = auc(roc_obj)
AUROC

#####################################################
############ MODEL 5###############################
####################################################
#Model 5 is built based on all variables (numeric and categorical).


predictors = c(1,2,3,4,5,6,7,8,9,10,11,12)  # index of the columns of predictors

min_leaf_size <- 5 # stopping criteria : Stop when no subregion contains more than five observations
max_depth <- 4 # 3,4,6, 10,... etc max depth  of decision tree

tree <- built_tree(train[, predictors], train[, 13])
#Print Rules decision
print_node(tree)

# Plot Tree
plot_tree(tree)

# To get the prediction on test set given decision rules
testdata <-cbind(test[ ,predictors],test[, 13])
prediction <- predict_dt(testdata[,1:(ncol(testdata)-1)],tree)

# To get the accuracy of the prediction (percentage) ChurnerPredicted/ChurnerReal
res <- cbind(test[, 13], as.numeric(prediction))
length(which(res[,1] == res[,2])) / nrow(testdata)

# Type Error 1 - Type Error 2
pred_ident <- table(apply(res, 1, pred_type_row))
true_pos_rate <- pred_ident[4] / (pred_ident[4] + pred_ident[1])
false_pos_rate <- pred_ident[2] / (pred_ident[3] + pred_ident[2])
true_pos_rate
false_pos_rate


#ROC CURVE
plot.roc(test[,13], as.numeric(prediction), legacy.axes = T, percent = T)
# AUROC
roc_obj<- roc(test[,13], as.numeric(prediction))
AUROC = auc(roc_obj)
AUROC




##########################################

##### DECISION TREE - CROSS VALIDATION ####
####### SEQUENTIAL CROSS VALIDATION #######

#Set random seed. Don't remove this line.
set.seed(195)
# Shuffle the dataset; build train and test ,Don't remove this line.
n <- nrow(churn.data)
shuffled <- churn.data[sample(n),] #Don't remove this line.
kfold=10 ##Don't remove this line.

#####################################################
############ MODEL 1 ###############################
####################################################
# Model 1 : Considering all variables continuous 
#* The first model is built based on numeric variables: Credit Score,Tenure,Balance,Num Of Products, 
#Estimated Salary, Age and Credit Score given Age.

predictors = c(1,2,3,4,5,6,7) # index of the columns of continuous variables 
AUROC<-rep(0,kfold)
min_leaf_size <- 5 # stopping criteria : Stop when no subregion contains more than five observations
max_depth <- 10 # 6, 10,... etc max depth  of decision tree

for (i in 1:kfold) {
  indices <- (((i-1) * round((1/kfold)*nrow(shuffled))) + 1):((i*round((1/kfold) * nrow(shuffled))))
  # Exclude them from the train set
  train <- shuffled[-indices,]
  # Include them in the test set
  test <- shuffled[indices,]
  
  #Predict by Logistic Regression
  ntrain = dim(train)[1]
  ntest=dim(test)[1] 
  
  tree <- built_tree(train[, predictors], train[, 13])
  #print_node(tree)
  
  # To get the prediction on test set given decision rules
  testdata <-cbind(test[ ,predictors],test[, 13])
  prediction <- predict_dt(testdata[,1:(ncol(testdata)-1)],tree)
  roc_obj<- roc(testdata[,ncol(testdata)], as.numeric(prediction))
  AUROC[i] = auc(roc_obj)
}

## Summary Performance of Regression Logistic ##
# Mean of Error Type 1, 2 and AUROC
AvgAUROC<-mean(AUROC)
AvgAUROC 

# To get the index of Kfold where we got the maximum AUROC
maxAUROC = which.max(AUROC)
maxAUROC
AUROC[maxAUROC] 


#####################################################
############ MODEL 2 ###############################
####################################################

# Model 2 : Considering all categorical variables:Geography, Gender, HasCrCard and IsActiveMember.

predictors = c(8,9,10,11,12) # index of the columns of categorical variables
AUROC<-rep(0,kfold)
min_leaf_size <- 5 # stopping criteria : Stop when no subregion contains more than five observations
max_depth <- 10 # 6, 10,... etc max depth  of decision tree

for (i in 1:kfold) {
  indices <- (((i-1) * round((1/kfold)*nrow(shuffled))) + 1):((i*round((1/kfold) * nrow(shuffled))))
  # Exclude them from the train set
  train <- shuffled[-indices,]
  # Include them in the test set
  test <- shuffled[indices,]
  
  #Predict by Logistic Regression
  ntrain = dim(train)[1]
  ntest=dim(test)[1] 
  
  tree <- built_tree(train[, predictors], train[, 13])
  #print_node(tree)
  
  # To get the prediction on test set given decision rules
  testdata <-cbind(test[ ,predictors],test[, 13])
  prediction <- predict_dt(testdata[,1:(ncol(testdata)-1)],tree)
  roc_obj<- roc(testdata[,ncol(testdata)], as.numeric(prediction))
  AUROC[i] = auc(roc_obj)
}

## Summary Performance of Regression Logistic ##
# Mean of Error Type 1, 2 and AUROC
AvgAUROC<-mean(AUROC)
AvgAUROC 

# To get the index of Kfold where we got the maximum AUROC
maxAUROC = which.max(AUROC)
maxAUROC
AUROC[maxAUROC] 



#####################################################
############ MODEL 3 ###############################
####################################################
#Model 3 is built, using the most significant variables associated asymptotic p-values from t-tests.
#Age, Tenure, EstimatedSalary and HasaCard don't significant predictors.

predictors = c(1,4,5,7,8,9,10,12) # index of the columns of predictors
AUROC<-rep(0,kfold)
min_leaf_size <- 5 # stopping criteria : Stop when no subregion contains more than five observations
max_depth <- 10 # 6, 10,... etc max depth  of decision tree

for (i in 1:kfold) {
  indices <- (((i-1) * round((1/kfold)*nrow(shuffled))) + 1):((i*round((1/kfold) * nrow(shuffled))))
  # Exclude them from the train set
  train <- shuffled[-indices,]
  # Include them in the test set
  test <- shuffled[indices,]
  
  #Predict by Logistic Regression
  ntrain = dim(train)[1]
  ntest=dim(test)[1] 
  
  tree <- built_tree(train[, predictors], train[, 13])
  #print_node(tree)
  
  # To get the prediction on test set given decision rules
  testdata <-cbind(test[ ,predictors],test[, 13])
  prediction <- predict_dt(testdata[,1:(ncol(testdata)-1)],tree)
  roc_obj<- roc(testdata[,ncol(testdata)], as.numeric(prediction))
  AUROC[i] = auc(roc_obj)
}

## Summary Performance of Regression Logistic ##
# Mean of Error Type 1, 2 and AUROC
AvgAUROC<-mean(AUROC)
AvgAUROC 

# To get the index of Kfold where we got the maximum AUROC
maxAUROC = which.max(AUROC)
maxAUROC
AUROC[maxAUROC] 


#####################################################
############ MODEL 4 ###############################
####################################################
#Model 3 is built, using the most significant variables associated asymptotic p-values from t-tests.
#Age, Tenure, EstimatedSalary and HasaCard don't significant predictors.

predictors = c(2,4,8,9,10,12)  # index of the columns of categorical 
AUROC<-rep(0,kfold)
min_leaf_size <- 5 # stopping criteria : Stop when no subregion contains more than five observations
max_depth <- 10 # 6, 10,... etc max depth  of decision tree

for (i in 1:kfold) {
  indices <- (((i-1) * round((1/kfold)*nrow(shuffled))) + 1):((i*round((1/kfold) * nrow(shuffled))))
  # Exclude them from the train set
  train <- shuffled[-indices,]
  # Include them in the test set
  test <- shuffled[indices,]
  
  #Predict by Logistic Regression
  ntrain = dim(train)[1]
  ntest=dim(test)[1] 
  
  tree <- built_tree(train[, predictors], train[, 13])
  #print_node(tree)
  
  # To get the prediction on test set given decision rules
  testdata <-cbind(test[ ,predictors],test[, 13])
  prediction <- predict_dt(testdata[,1:(ncol(testdata)-1)],tree)
  roc_obj<- roc(testdata[,ncol(testdata)], as.numeric(prediction))
  AUROC[i] = auc(roc_obj)
}

## Summary Performance of Regression Logistic ##
# Mean of Error Type 1, 2 and AUROC
AvgAUROC<-mean(AUROC)
AvgAUROC 

# To get the index of Kfold where we got the maximum AUROC
maxAUROC = which.max(AUROC)
maxAUROC
AUROC[maxAUROC] 



#####################################################
############ MODEL 5###############################
####################################################
#Model 5 is built based on all variables (numeric and categorical).


predictors = c(1,2,3,4,5,6,7,8,9,10,11,12)  # index of the columns of predictors

AUROC<-rep(0,kfold)
min_leaf_size <- 5 # stopping criteria : Stop when no subregion contains more than five observations
max_depth <- 10 # 6, 10,... etc max depth  of decision tree

for (i in 1:kfold) {
  indices <- (((i-1) * round((1/kfold)*nrow(shuffled))) + 1):((i*round((1/kfold) * nrow(shuffled))))
  # Exclude them from the train set
  train <- shuffled[-indices,]
  # Include them in the test set
  test <- shuffled[indices,]
  
  #Predict by Logistic Regression
  ntrain = dim(train)[1]
  ntest=dim(test)[1] 
  
  tree <- built_tree(train[, predictors], train[, 13])
  #print_node(tree)
  
  # To get the prediction on test set given decision rules
  testdata <-cbind(test[ ,predictors],test[, 13])
  prediction <- predict_dt(testdata[,1:(ncol(testdata)-1)],tree)
  roc_obj<- roc(testdata[,ncol(testdata)], as.numeric(prediction))
  AUROC[i] = auc(roc_obj)
}

## Summary Performance of Regression Logistic ##
# Mean of Error Type 1, 2 and AUROC
AvgAUROC<-mean(AUROC)
AvgAUROC 

# To get the index of Kfold where we got the maximum AUROC
maxAUROC = which.max(AUROC)
maxAUROC
AUROC[maxAUROC] 


