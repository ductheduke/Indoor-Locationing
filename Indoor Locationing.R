# Title: Indoor Location

# Last update: 1.15.2020

# File/project name: C3T3.R
# RStudio Project name: WLAN Indoor Positioning

#################
# Project Notes #
#################

# Summarize project: The purpose of this project is to predict the location 
# based on the WiFi imprints

################
# Housekeeping #
################

# Clear objects if necessary
rm(list = ls())

####################
# Install packages #
####################

install.packages("caret", dependencies = c("Depends", "Suggests"))
install.packages("corrplot")
install.packages("readr")
install.packages("lattice")
install.packages("ggplot2")
install.packages("inum")
install.packages("FactoMineR")
install.packages("doParallel")
install.packages("leaps")
install.packages("prodlim")
install.packages("ipred")
install.packages("mlbench")
install.packages("randomForest")
install.packages("kknn")
install.packages("igraph")
install.packages("C50")
install.packages("mvtnorm")
install.packages("libcoin")
install.packages("inum")
install.packages("partykit")
install.packages("ISLR")


#################
# Load packages #
#################

library(dplyr)
library(ISLR)
library(lattice)
library(ggplot2)
library(caret)
library(tidyr)


###############
# Import data #
###############

#### --- Load raw/preprocessed datasets --- ####

trainingSet <- read.csv("trainingdata.csv", stringsAsFactors = FALSE, check.names = TRUE)


#################
# Evaluate data #
#################

summary(trainingSet[,521:529])
# check for missing values 
any(is.na(trainingSet)) # no missing values
# Plot Longtitude and Latitude
plot(trainingSet$LONGITUDE, trainingSet$LATITUDE)


######################
# Data Visualization #
######################

ggplot(trainingSet, aes(LONGITUDE,
                     LATITUDE),
       colour = BUILDINGID) +
  ggtitle("Building ID by Longitude and Latitude") +
  geom_hex() +
  theme(legend.position = "bottom")


##################
# Pre-Processing #
##################

# Remove the ID attributes
trainingSet$USERID <- NULL
trainingSet$PHONEID <- NULL
trainingSet$TIMESTAMP <- NULL
trainingSet$RELATIVEPOSITION <- NULL

# Remove Longitude and Latitude because they are not necessary for predicting indoor location
trainingSet$LONGITUDE <- NULL
trainingSet$LATITUDE <- NULL

# Convert BuildingID, Floor and Space ID to factors
trainingSet$FLOOR <- as.factor(trainingSet$FLOOR)
trainingSet$SPACEID <- as.factor(trainingSet$SPACEID)
trainingSet$BUILDINGID <- as.factor(trainingSet$BUILDINGID)

summary(trainingSet[,521:523])


#######################
# Feature Engineering #
#######################

# Create a unique location ID that combines both Floor and SpaceID
trainingSet$LOC_ID <- paste(trainingSet$FLOOR, trainingSet$SPACEID, sep="_")
# Remove both Floor and SpaceID
trainingSet$FLOOR <- NULL
trainingSet$SPACEID <- NULL
# Confirm the structure of new data frame
summary(trainingSet[,521:522])
# Save as the training Set
train <- trainingSet
write.csv(train, file="C3T3preproc.csv")


##############
# Subsetting #
##############

# Subset the dataset by Building ID
B0 <- filter(train, BUILDINGID == 0)
B1 <- filter(train, BUILDINGID == 1)
B2 <- filter(train, BUILDINGID == 2)

# Remove Building ID from all 3 subsets
B0$BUILDINGID <- NULL
B1$BUILDINGID <- NULL
B2$BUILDINGID <- NULL

# Convert LOC_ID to factor again and droplevel
B0$LOC_ID <- as.factor(B0$LOC_ID)
B1$LOC_ID <- as.factor(B1$LOC_ID)
B2$LOC_ID <- as.factor(B2$LOC_ID)

B0$LOC_ID <- droplevels(B0$LOC_ID)
B1$LOC_ID <- droplevels(B1$LOC_ID)
B2$LOC_ID <- droplevels(B2$LOC_ID)

summary(B0[,520:521])
summary(B1[,520:521])
summary(B2[,520:521])


####################
# Training Control #
####################

fitCtrl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
seed <- 123


###################
# Train/Test Sets #
###################

# ---- Building 0 ----#
# create the training partition that is 75% of total obs
set.seed(seed)
inTraining <- createDataPartition(B0$LOC_ID, p=0.75, list=FALSE)
B0_train <- B0[inTraining,]   
B0_test <- B0[-inTraining,]   
nrow(B0_train)  
nrow(B0_test) 

# ---- Building 1 ----#
# create the training partition that is 75% of total obs
set.seed(seed)
inTraining <- createDataPartition(B1$LOC_ID, p=0.75, list=FALSE)
B1_train <- B1[inTraining,]   
B1_test <- B1[-inTraining,]   
nrow(B1_train)  
nrow(B1_test) 

# ---- Building 2 ----#
# create the training partition that is 75% of total obs
set.seed(seed)
inTraining <- createDataPartition(B2$LOC_ID, p=0.75, list=FALSE)
B2_train <- B2[inTraining,]   
B2_test <- B2[-inTraining,]   
nrow(B2_train)  
nrow(B2_test) 


###############
# Train model #
###############

# ------- k-Nearest Neighbor ------- #
#--- Building 0 ---#
set.seed(seed)
knn_B0 <- train(LOC_ID ~ ., 
                data = B0_train,
                method = "knn",
                trControl = fitCtrl,
                preProcess = c("zv", "medianImpute"))
knn_B0
knn_B0_Summary <- capture.output(knn_B0)
cat("Summary", knn_B0_Summary,
    file = "summary of knn_B0.txt",
    sep = "\n",
    append = TRUE)

#--- Building 1 ---#
set.seed(seed)
knn_B1 <- train(LOC_ID ~ ., 
                data = B1_train,
                method = "knn",
                trControl = fitCtrl,
                preProcess = c("zv", "medianImpute"))
knn_B1
knn_B1_Summary <- capture.output(knn_B1)
cat("Summary", knn_B1_Summary,
    file = "summary of knn_B1.txt",
    sep = "\n",
    append = TRUE)

#--- Building 2 ---#
set.seed(seed)
knn_B2 <- train(LOC_ID ~ ., 
                data = B2_train,
                method = "knn",
                trControl = fitCtrl,
                preProcess = c("zv", "medianImpute"))
knn_B2
knn_B2_Summary <- capture.output(knn_B2)
cat("Summary", knn_B2_Summary,
    file = "summary of knn_B2.txt",
    sep = "\n",
    append = TRUE)


# ------- Random Forest ------- #
rfGrid <- expand.grid(mtry = seq(1, 10, by = 1))
# best mtry = 10 after running the above Grid so will use this as the tuning parameter below to save time

#--- Building 0 ---#
set.seed(seed)
rf_B0 <- train(LOC_ID ~ ., 
               data = B0_train,
               method = "rf",
               tuneGrid=rfGrid,
               trControl = fitCtrl,
               preProcess = c("zv", "medianImpute"))
rf_B0
rf_B0_Summary <- capture.output(rf_B0)
cat("Summary", rf_B0_Summary,
    file = "summary of rf_B0.txt",
    sep = "\n",
    append = TRUE)

#--- Building 1 ---#
set.seed(seed)
rf_B1 <- train(LOC_ID ~ ., 
               data = B1_train,
               method = "rf",
               tuneGrid=data.frame(mtry=10),
               trControl = fitCtrl,
               preProcess = c("zv", "medianImpute"))
rf_B1
rf_B1_Summary <- capture.output(rf_B1)
cat("Summary", rf_B1_Summary,
    file = "summary of rf_B1.txt",
    sep = "\n",
    append = TRUE)

#--- Building 2 ---#
set.seed(seed)
rf_B2 <- train(LOC_ID ~ ., 
               data = B2_train,
               method = "rf",
               tuneGrid=data.frame(mtry=10),
               trControl = fitCtrl,
               preProcess = c("zv", "medianImpute"))
rf_B2
rf_B2_Summary <- capture.output(rf_B2)
cat("Summary", rf_B2_Summary,
    file = "summary of rf_B2.txt",
    sep = "\n",
    append = TRUE)


# ------- C5.0 ------- #
#--- Building 0 ---#
set.seed(seed)
c50_B0 <- train(LOC_ID ~ ., 
                data = B0_train,
                method = "C5.0",
                trControl = fitCtrl,
                preProcess = c("zv", "medianImpute"))
c50_B0
c50_B0_Summary <- capture.output(c50_B0)
cat("Summary", c50_B0_Summary,
    file = "summary of c50_B0.txt",
    sep = "\n",
    append = TRUE)

#--- Building 1 ---#
set.seed(seed)
c50_B1 <- train(LOC_ID ~ ., 
                data = B1_train,
                method = "C5.0",
                trControl = fitCtrl,
                preProcess = c("zv", "medianImpute"))
c50_B1
c50_B1_Summary <- capture.output(c50_B1)
cat("Summary", c50_B1_Summary,
    file = "summary of c50_B1.txt",
    sep = "\n",
    append = TRUE)

#--- Building 2 ---#
set.seed(seed)
c50_B2 <- train(LOC_ID ~ ., 
                data = B2_train,
                method = "C5.0",
                trControl = fitCtrl,
                preProcess = c("zv", "medianImpute"))
c50_B2
c50_B2_Summary <- capture.output(c50_B2)
cat("Summary", c50_B2_Summary,
    file = "summary of c50_B2.txt",
    sep = "\n",
    append = TRUE)


###################
# Evaluate models #
###################

Model_B0 <- resamples(list(rf=rf_B0, knn=knn_B0, c50=c50_B0))
summary(Model_B0)
Model_B1 <- resamples(list(rf=rf_B1, knn=knn_B1, c50=c50_B1))
summary(Model_B1)
Model_B2 <- resamples(list(rf=rf_B2, knn=knn_B2, c50=c50_B2))
summary(Model_B2)
#--- Conclusion ---#
# Top model for predicting Floor is: rf with mtry = 10


######################
# Validate top model #
######################

# Create Train and Test data frames that include all buildings
set.seed(seed)
inTraining <- createDataPartition(train$LOC_ID, p=0.75, list=FALSE)
trainSet <- train[inTraining,]   
testSet <- train[-inTraining,]   
nrow(trainSet)  
nrow(testSet)

# Remove the Building ID attribute from both data frames
trainSet$BUILDINGID <- NULL
testSet$BUILDINGID <- NULL

# Make sure LOC_ID is Factor
trainSet$LOC_ID <- as.factor(trainSet$LOC_ID)
testSet$LOC_ID <- as.factor(testSet$LOC_ID)

# Confirm structure of both the Train and Test sets
summary(trainSet[,520:521])
summary(testSet[,520:521])

# Create random sample of the trainSet and testSet
sample_trainSet <- trainSet[sample(1:nrow(trainSet), 10000, replace = FALSE),]
sample_testSet <- testSet[sample(1:nrow(testSet), 3000, replace = FALSE),]
nrow(sample_trainSet)
nrow(sample_testSet)
summary(sample_trainSet[,520:521])

# Drop level LOC_ID in both trainSet and testSet
sample_trainSet$LOC_ID <- droplevels(sample_trainSet$LOC_ID)
sample_testSet$LOC_ID <- droplevels(sample_testSet$LOC_ID)

# Create the RF Model to predict LOC_ID
set.seed(seed)
rf_LOC <- train(LOC_ID ~ ., 
               data = sample_trainSet,
               method = "rf",
               tuneGrid=data.frame(mtry=10),
               trControl = fitCtrl,
               preProcess = c("zv", "medianImpute"))
rf_LOC
# Accuracy = 0.5011716
# Kappa = 0.4991368
rf_LOC_Summary <- capture.output(rf_LOC)
cat("Summary", rf_LOC_Summary,
    file = "summary of rf_LOC.txt",
    sep = "\n",
    append = TRUE)
# Make Predictions
rfPred_LOC <- predict(rf_LOC, sample_testSet)
# Performance Metrics
PredLOC <- postResample(rfPred_LOC, sample_testSet$LOC_ID)
PredLOC
# Accuracy = 0.5016667
# Kappa = 0.4997239
PredLOC_Summary <- capture.output(PredLOC)
cat("Summary", PredLOC_Summary,
    file = "summary of PredLOC.txt",
    sep = "\n",
    append = TRUE)
