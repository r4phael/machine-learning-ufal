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

createDf <- function (){
  results <<-
    data.frame(
      Project = character(),
      Algorithm = character(),
      Precision = character(),
      Recall = character(),
      "F-Measure" = character()
    )
}

finalResults <- function(resultsAlgo, project, algo){
  results <<-
    rbind(
      results,
      data.frame(
        "Project" = project,
        "Algorithm" = algo,
        "Precision" = median(sapply(resultsAlgo, "[[", 1)),
        "Recall" = median(sapply(resultsAlgo, "[[", 2)),
        "F-Measure" = median(sapply(resultsAlgo, "[[", 3))
      )
    )
}



#####################################################################################################
#                               Results Balanced                                                    #  
#####################################################################################################

filenames = list.files(path = "/home/r4ph/R/machine-learning-ufal/datasets/vulnerability/balanced",
                       full.names = TRUE,
                       recursive = TRUE)

projects <- c("GlibC", "Httpd", "Kernel", "Mozilla", "Xen")
setwd("/home/r4ph/R/machine-learning-ufal/datasets/vulnerability/results/")

#Create data frame:
createDf()

#Apply Results Balanced
for (i in 1:length(filenames)) {
  
  cat("Writting ")
  
  dataset <- read.csv(filenames[i])
  #project <- strsplit(filenames, split = '/')[[i]][9]
  folds <- createFolds(dataset[1:27], k = 10, returnTrain = TRUE)
  
  rows <- nrow(dataset)
  cat(paste0("Input: ", rows, " rows in project... ", projects[i], "\n"))
  
  #Results C50
  finalResults(executeC50(dataset, folds), projects[i], "C50")
  #Results Bayes
  finalResults(executeNaiveBayes(dataset, folds), projects[i], "NaiveBayes")
  #Algorithm SVM
  finalResults(executeSVM(dataset, folds), projects[i], "SVM")
  #Jr48
  finalResults(executeJ48(dataset, folds), projects[i], "Jr48")
  #OneR
  finalResults(executeOneR(dataset, folds), projects[i], "OneR")
  #RandomForest
  finalResults(executeRandomForest(dataset, folds), projects[i], "RandomForest")
  
}

write.csv(x =  results, file = "ml_balanced.csv")



#####################################################################################################
#                               Results With Features Selection                                     #  
#####################################################################################################

#Create data frame:
createDf()

#Apply Results Balanced
for (i in 1:length(filenames)) {
  
  features <- makeClassifTask(id = deparse(substitute(dataset)), dataset, colnames(dataset)[28] )
  fv = generateFilterValuesData(features, method = "information.gain")
  filtered = filterFeatures(features, fval = fv, threshold = 0.2)
  
  filtered
  
  cat("Writting ")
  
  dataset <- read.csv(filenames[i])
  #project <- strsplit(filenames, split = '/')[[i]][9]
  folds <- createFolds(dataset[1:27], k = 10, returnTrain = TRUE)
  
  rows <- nrow(dataset)
  cat(paste0("Input: ", rows, " rows in project... ", projects[i], "\n"))
  
  #Results C50
  finalResults(executeC50(dataset, folds), projects[i], "C50")
  #Results Bayes
  finalResults(executeNaiveBayes(dataset, folds), projects[i], "NaiveBayes")
  #Algorithm SVM
  finalResults(executeSVM(dataset, folds), projects[i], "SVM")
  #Jr48
  finalResults(executeJ48(dataset, folds), projects[i], "Jr48")
  #OneR
  finalResults(executeOneR(dataset, folds), projects[i], "OneR")
  #RandomForest
  finalResults(executeRandomForest(dataset, folds), projects[i], "RandomForest")
  
}

#Finals Results Unbal:
write.csv(x = results, file = "ml_features.csv")




#####################################################################################################
#                               Results Unbalanced                                                    #  
#####################################################################################################

filenamesUnbal = list.files(path = "/home/r4ph/R/machine-learning-ufal/datasets/vulnerability/unbalanced",
                       full.names = TRUE,
                       recursive = TRUE)

#Create data frame:
results <<-
  data.frame(
    Project = character(),
    Algorithm = character(),
    Precision = character(),
    Recall = character(),
    "F-Measure" = character()
  )


#Apply Results Balanced
for (i in 1:length(filenames)) {
  
  cat("Writting ")
  
  dataset <- read.csv(filenames[i])
  #project <- strsplit(filenames, split = '/')[[i]][9]
  folds <- createFolds(dataset[1:27], k = 10, returnTrain = TRUE)
  
  rows <- nrow(dataset)
  cat(paste0("Input: ", rows, " rows in project... ", projects[i], "\n"))
  
  #Results C50
  finalResults(executeC50(dataset, folds), projects[i], "C50")
  #Results Bayes
  finalResults(executeNaiveBayes(dataset, folds), projects[i], "C50")
  #Algorithm SVM
  finalResults(executeSVM(dataset, folds), projects[i], "C50")
  #Jr48
  finalResults(executeJ48(dataset, folds), projects[i], "C50")
  #OneR
  finalResults(executeOneR(dataset, folds), projects[i], "C50")
  #RandomForest
  finalResults(executeRandomForest(dataset, folds), projects[i], "C50")
  
}

#Finals Results Unbal:
write.csv(x = results, file = "ml_unbalanced.csv")

