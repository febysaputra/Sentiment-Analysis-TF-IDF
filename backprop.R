library("neuralnet")

getAccuracy <- function(cf){
  diag <- 0
  div <- 0
  for (i in 1:nrow(cf)){
    for (j in 1:ncol(cf)){
      if(i==j){
        diag = diag + cf[i,j]
      }
      div = div + cf[i,j]
    } 
  }
  return(diag/div)
}

set.seed(1234)

data <- tfidf_LVQ
data <- data[sample(nrow(data)),]
ind <- sample(2, nrow(data), replace=TRUE, prob =c(0.8,0.2))
trainData <- data[ind==1,]
testData <- data[ind==2,]

f <- as.formula(paste("kelas_manual", paste(names(trainData[-ncol(trainData)]), collapse=" + "), sep=" ~ "))
net <- neuralnet(f, trainData, hidden=4, rep=50, learningrate = 0.1, algorithm = 'backprop', linear.output = FALSE)
pred <- predict(net, testData[-ncol(testData)])     
cf <- table(testData$kelas_manual, apply(pred, 1, which.max))
getAccuracy(cf)
