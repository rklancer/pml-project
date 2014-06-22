# load libraries, etc.
# make sure to load and register doMC

require(ggplot2)
require(caret)
require(doMC)

# tell caret to use parallel processing when tuning parameters
registerDoMC(cores=4)

# load in the data
# note that pml-testing.csv is NOT a "testing set" for purposes
# of estimating out-of-sample error
data <- read.csv("data/pml-training.csv", na.strings=c("NA", ""))

# set seed so test/train split remains same.
set.seed(62412)

# split test/train
inTrain <- createDataPartition(y=data$classe, p=0.7, list=FALSE)
rawTraining <- data[inTrain,]
rawTesting <- data[-inTrain,]

# examine columns for NAs
numNA <- sapply(rawTraining, function(col) { sum(is.na(col))/length(col) })
table(round(numNA, digits=2))
keep <- numNA < 0.1

# let's take a look at the "leakage" variables
qplot(rawTraining$X, rawTraining$classe, color=rawTraining$user)
qplot(training$raw_timestamp_part_1, training$classe, color=training$user, )
qplot(training$raw_timestamp_part_2, training$classe, color=training$user, )
qplot(training$cvtd_timestamp, training$classe, color=training$user, )
qplot(rawTraining$num_window, rawTraining$classe, color=rawTraining$user)

keep[["X"]] <- FALSE
keep[grep("timestamp", names(keep))] <- FALSE
keep[["num_window"]] <- FALSE

#  new_window is perfectly correlated with NA values, example:
cor(is.na(rawTraining[,which(numNA > 0.1)[1]]), unclass(rawTraining$new_window))
keep[["new_window"]] <- FALSE

# keep only the columns with a low fraction of NAs, convert integer
# columns to numerics. This should be applicable to both test dataset
# and assignment-evaluation dataset in "data/pml-testing.csv"
prep <- function (data) {
    data <- data[,keep]
    data <- data.frame(lapply(data, function (col) {
        if (class(col) == "integer") {
            col <- as.numeric(col)
        }
        col
    }))
    data
}
    
training <- prep(rawTraining)

# Plot correlations, out of curiosity
m <- abs(cor(sapply(training, unclass)))
diag(m) <- 0
m <- data.frame(m)

# there are a handful of highly-correlated columns -- but not many
hist(as.vector(as.matrix(m)))

# not many columns very highly correlated with classe
hist(m$classe)

# debug the method (incl. usage of oob, parallel)  using 10% of the training data

getSubset <- function(p=0.1) {
    inSubData <- createDataPartition(training$classe, p=p, list=FALSE)
    subData <- training[inSubData,]
    
    inSubTrain <- createDataPartition(subData$classe, p=0.7, list=FALSE)
    subTraining <- subData[inSubTrain,]
    subTesting <- subData[-inSubTrain,]
    list(training=subTraining, testing=subTesting)
}
    
doPrediction <- function(training, testing) {
    trControl = trainControl(method="oob")
    time <- system.time(fit <- train(classe ~ ., data=training, method="rf", trControl=trControl))
    predictions <- predict(fit, newdata=testing)
    confmat <- confusionMatrix(data=predictions, reference=testing$classe)
    list(time=time, fit=fit, predictions=predictions, confmat=confmat)
}

# when applying doPrediction to testing data, remember to prep:
testing <- prep(rawTesting)
results <- doPrediction(training, testing)
# (examine results)
