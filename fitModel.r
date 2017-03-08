# Logistic regression model to predict cancer
cancer = read.csv("cancerSubset15.csv", header = F, sep = ",")
names(cancer) = c("hasCancer", "health", "hypertension", "bloodCholesterol", "heartAttack","CHD",
                  "stroke", "asthma", "COPD", "arthritis", "depression", "kidney", "diabetes")
cancer = cancer[complete.cases(cancer),]
cancer.scale = cancer[(cancer[1] == 1 | cancer[1] == 2) &
                        (cancer[2] == 1 | cancer[2] == 2 | cancer[2] == 3 | cancer[2] == 4 | cancer[2] == 5) &
                        (cancer[3] == 1 | cancer[3] == 3) &
                        (cancer[4] == 1 | cancer[4] == 2) &
                        (cancer[5] == 1 | cancer[5] == 2) &
                        (cancer[6] == 1 | cancer[6] == 2) &
                        (cancer[7] == 1 | cancer[7] == 2) &
                        (cancer[8] == 1 | cancer[8] == 2) &
                        (cancer[9] == 1 | cancer[9] == 2) &
                        (cancer[10] == 1 | cancer[10] == 2) &
                        (cancer[11] == 1 | cancer[11] == 2) &
                        (cancer[12] == 1 | cancer[12] == 2) &
                        (cancer[13] == 1 | cancer[13] == 3),
                      ]
cancer.scale[cancer.scale[,1] == 2,1] = 0
cancer.scale[,1] = as.numeric(cancer.scale[,1])
glm.out<-glm(hasCancer ~ health+hypertension+bloodCholesterol+heartAttack+CHD+
             stroke+asthma+COPD+arthritis+depression+kidney+diabetes, data=cancer.scale, family=binomial(link=logit))

summary(glm.out)

# Visualize in ROC
prob=predict(glm.out,type=c("response"))
cancer.scale$LRprob=prob
library(pROC)
lrplot <- roc(hasCancer ~ prob, data = cancer.scale)
plot(lrplot)

# SVM model to predict cancer
cancer.scale.sub = cancer.scale[1:10000,] #subset the data for speed

# Tune SVM model
library(e1071)
svm.tune = tune(svm, hasCancer ~ health+hypertension+bloodCholesterol+heartAttack+CHD+
                  stroke+asthma+COPD+arthritis+depression+kidney+diabetes, data=cancer.scale.sub,
                kernel="radial", ranges=list(cost=c(0.2,1,5), gamma=c(0.2,1,5)))
print(svm.tune)

# Fit SVM model
svm.model <- svm(hasCancer ~ health+hypertension+bloodCholesterol+heartAttack+CHD+
                   stroke+asthma+COPD+arthritis+depression+kidney+diabetes, data=cancer.scale.sub,
                 cost = 1, gamma = 1)
summary(svm.model)

# Visualize in ROC
prob=predict(svm.model,type=c("response"))
cancer.scale.sub$SVMprob=prob
library(pROC)
svmplot <- roc(hasCancer ~ SVMprob, data = cancer.scale.sub)
plot(svmplot,col = "blue", add = TRUE)
