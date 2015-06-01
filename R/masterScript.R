setwd("C:/Users/Victor/Desktop/digitKaggle/R")

library(optimx)

source("sigmoid.R")

trainSet <- as.matrix(read.csv("C:/Users/Victor/Desktop/digitKaggle/train.csv", header = TRUE))

y <- as.matrix(trainSet[,1])

X <- trainSet[,2:ncol(trainSet)]

