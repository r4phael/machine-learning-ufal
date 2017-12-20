#Program: Contains machine learning algorithms and some functions to calculate precision, recall and f_measure.
#Input:
#Output

library(RWeka)
library(e1071)
library(gmodels)
library(C50)
library(caret)
library(irr)
library(randomForest)



# Function to calculate precision
precision <- function(tp, fp) {
  precision <- tp / (tp + fp)
  
  return(precision)
}

# Function to calculate recall
recall <- function(tp, fn) {
  recall <- tp / (tp + fn)
  
  return(recall)
}

# Function to calculate F-measure
f_measure <- function(tp, fp, fn) {
  f_measure <-
    (2 * precision(tp, fp) * recall(tp, fn)) / (recall(tp, fn) + precision(tp, fp))
  
  return(f_measure)
}

# Function to calculate true_positive, true_negative, false_positive, false_negative
measures <- function(test, pred) {
  true_positive <- 0
  true_negative <- 0
  false_positive <- 0
  false_negative <- 0
  
  for (i in 1:length(pred)) {
    if (test[i] == 'VULNERABLE' && pred[i] == 'VULNERABLE') {
      true_positive <- true_positive + 1
    } else if (test[i] == 'NEUTRAL' && pred[i] == 'NEUTRAL') {
      true_negative <- true_negative + 1
    } else if (test[i] == 'NEUTRAL' && pred[i] == 'VULNERABLE') {
      false_negative <- false_negative + 1
    } else if (test[i] == 'VULNERABLE' && pred[i] == 'NEUTRAL') {
      false_positive <- false_positive + 1
    }
  }
  
  measures <-
    c(
      precision(true_positive, false_positive),
      recall(true_positive, false_negative),
      f_measure(true_positive, false_positive, false_negative)
    )
  
  return(measures)
}

# Techiniques
executeJ48 <- function(dataset, folds) {
  results <- lapply(folds, function(x) {
    train <- dataset[-x,]
    test <- dataset[x,]
    model <- J48(train$Affected ~ ., data = train)
    pred <- predict(model, test)
    results <- measures(test$Affected, pred)
    
    return(results)
  })
  
}

executeNaiveBayes <- function(dataset, folds) {
  results <- lapply(folds, function(x) {
    train <- dataset[-x,]
    test <- dataset[x,]
    model <- naiveBayes(train, train$Affected, laplace = 1)
    pred <- predict(model, test)
    
    results <- measures(test$Affected, pred)
    
    return(results)
  })
  
}

executeSVM <- function(dataset, folds) {
  results <- lapply(folds, function(x) {
    train <- dataset[-x, ]
    test <- dataset[x, ]
    model <- svm(train$Affected ~ ., data = train)
    pred <- predict(model, test)
    
    results <- measures(test$Affected, pred)
    
    return(results)
  })
  
}

executeOneR <- function(dataset, folds) {
  results <- lapply(folds, function(x) {
    train <- dataset[-x, ]
    test <- dataset[x, ]
    model <- OneR(train$Affected ~ ., data = train)
    pred <- predict(model, test)
    
    results <- measures(test$Affected, pred)
    
    return(results)
  })
  
}


executeRandomForest <- function(dataset, folds) {
  results <- lapply(folds, function(x) {
    train <- dataset[-x, ]
    test <- dataset[x, ]
    model <- randomForest(train$Affected ~ ., data = train)
    pred <- predict(model, test)
    
    results <- measures(test$Affected, pred)
    
    return(results)
  })
}


executeC50 <- function(dataset, folds) {
  results <- lapply(folds, function(x) {
    train <- dataset[-x, ]
    test <- dataset[x, ]
    model <- C5.0(train$Affected ~ ., data = train)
    pred <- predict(model, test)
    results <- measures(test$Affected, pred)
    return(results)
  })
  
}


#####################################################################################################
#                               Results Balanced                                                    #  
#####################################################################################################

filenames = list.files(path = "/home/r4ph/R/machine-learning-ufal/datasets/vulnerability/balanced",
                       full.names = TRUE,
                       recursive = TRUE)

#Create data frame:
resultsBal <-
  data.frame(
    Project = character(),
    Algorithm = character(),
    Precision = character(),
    Recall = character(),
    "F-Measure" = character()
  )


#Apply Results Balanced
for (i in 1:length(filenames)) {
  folds <- createFolds(elements, k = 10, returnTrain = TRUE)
  
  dataset <- read.csv(filenames[i], stringsAsFactors = FALSE)
  project <- strsplit(filenames, split = '/')[[i]][9]
  
  resultsC50 <- executeC50(elements, folds)
  resultsBayes <- executeNaiveBayes(elements, folds)
  resultsSVM <- executeSVM(elements, folds)
  #resultsJ48 <- executeJ48(elements, folds)
  #resultsOneR <- executeOneR(elements, folds)
  resultsRF <- executeRandomForest(elements, folds)
  
  resultsBal <-
    rbind(
      resultsBal,
      data.frame(
        "Project" = project,
        "Algorithm" = "C50",
        Precision = median(sapply(resultsC50, "[[", 1)),
        "Recall" = median(sapply(resultsC50, "[[", 2)),
        "F-Measure" = median(sapply(resultsC50, "[[", 3))
      )
    )
  
  resultsBal <-
    rbind(
      resultsBal,
      data.frame(
        "Project" = project,
        "Algorithm" = "Naive Bayes",
        Precision = median(sapply(resultsBayes, "[[", 1)),
        "Recall" = median(sapply(resultsBayes, "[[", 2)),
        "F-Measure" = median(sapply(resultsBayes, "[[", 3))
      )
    )
  
  resultsBal <-
    rbind(
      resultsBal,
      data.frame(
        "Project" = project,
        "Algorithm" = "SVM",
        Precision = median(sapply(resultsSVM, "[[", 1)),
        "Recall" = median(sapply(resultsSVM, "[[", 2)),
        "F-Measure" = median(sapply(resultsSVM, "[[", 3))
      )
    )
  
  resultsBal <-
    rbind(
      resultsBal,
      data.frame(
        "Project" = project,
        "Algorithm" = "Ramdom Forest",
        Precision = median(sapply(resultsRF, "[[", 1)),
        "Recall" = median(sapply(resultsRF, "[[", 2)),
        "F-Measure" = median(sapply(resultsRF, "[[", 3))
      )
    )
  
}


#####################################################################################################
#                               Results Unbalanced                                                    #  
#####################################################################################################

filenames = list.files(path = "/home/r4ph/R/machine-learning-ufal/datasets/vulnerability/unbalanced",
                       full.names = TRUE,
                       recursive = TRUE)

#Create data frame:
resultsUnbal <-
  data.frame(
    Project = character(),
    Algorithm = character(),
    Precision = character(),
    Recall = character(),
    "F-Measure" = character()
  )



for (i in 1:length(filenames)) {
  folds <- createFolds(elements, k = 10, returnTrain = TRUE)
  
  dataset <- read.csv(filenames[i], stringsAsFactors = FALSE)
  dataset$Fold <- NULL
  project <- strsplit(filenames, split = '/')[[i]][9]
  
  resultsC50 <- executeC50(elements, folds)
  resultsBayes <- executeNaiveBayes(elements, folds)
  resultsSVM <- executeSVM(elements, folds)
  resultsRF <- executeRandomForest(elements, folds)
  
  resultsUnbal <-
    rbind(
      resultsUnbal,
      data.frame(
        "Project" = project,
        "Algorithm" = "C50",
        Precision = median(sapply(resultsC50, "[[", 1)),
        "Recall" = median(sapply(resultsC50, "[[", 2)),
        "F-Measure" = median(sapply(resultsC50, "[[", 3))
      )
    )
  
  resultsUnbal <-
    rbind(
      resultsUnbal,
      data.frame(
        "Project" = project,
        "Algorithm" = "Naive Bayes",
        Precision = median(sapply(resultsBayes, "[[", 1)),
        "Recall" = median(sapply(resultsBayes, "[[", 2)),
        "F-Measure" = median(sapply(resultsBayes, "[[", 3))
      )
    )
  
  resultsUnbal <-
    rbind(
      resultsUnbal,
      data.frame(
        "Project" = project,
        "Algorithm" = "SVM",
        Precision = median(sapply(resultsSVM, "[[", 1)),
        "Recall" = median(sapply(resultsSVM, "[[", 2)),
        "F-Measure" = median(sapply(resultsSVM, "[[", 3))
      )
    )
  
  resultsUnbal <-
    rbind(
      resultsUnbal,
      data.frame(
        "Project" = project,
        "Algorithm" = "Ramdom Forest",
        Precision = median(sapply(resultsRF, "[[", 1)),
        "Recall" = median(sapply(resultsRF, "[[", 2)),
        "F-Measure" = median(sapply(resultsRF, "[[", 3))
      )
    )
  
}


#Finals Results Bal:
resultsBal

#Finals Results Unbal:
resultsUnbal


#Utilizar técnicas de seleção de atributos para saber qual atributo é o mais importante;
#
