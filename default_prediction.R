library(dplyr)
library(tidyr)
library(Hmisc)
library(caret)
library(corrplot)
library(Amelia)
library(randomForest)

#load data
d <- read.csv("~/Downloads/dataset.csv")

#  view the data

str(d)
summary(d)


missmap(d, main = "Missing values") # no missing values

# let's check the creditability of the people based on the parameters provided

# There are 700 creditable & 300 not creditable obs

table(d$Creditability)

# convert all the continuous variables to categorical variables


d$durationOfCreditCategorised <- ifelse(d$Duration.of.Credit..month.<=6,10,
                                        ifelse(d$Duration.of.Credit..month. <= 12,9,
                                               ifelse(d$Duration.of.Credit..month. <=18,8,
                                                      ifelse(d$Duration.of.Credit..month.<=24,7,
                                                             ifelse(d$Duration.of.Credit..month.<=30,6,
                                                                    ifelse(d$Duration.of.Credit..month.<=36,5,
                                                                           ifelse(d$Duration.of.Credit..month.<=42,4,
                                                                                  ifelse(d$Duration.of.Credit..month.<=48,3,
                                                                                         ifelse(d$Duration.of.Credit..month.<=54,2,1)))))))))


View(d)



table(d$durationOfCreditCategorised)
summary(d$Credit.Amount)




d$Credit.Amount.Categorised <- ifelse(d$Credit.Amount <=500,10,
                                      ifelse(d$Credit.Amount<=1000,9,
                                             ifelse(d$Credit.Amount<=1500,8,
                                                    ifelse(d$Credit.Amount<=2500,7,
                                                           ifelse(d$Credit.Amount<=5000,6,
                                                                  ifelse(d$Credit.Amount<=7500,5,
                                                                         ifelse(d$Credit.Amount<=10000,4,
                                                                                ifelse(d$Credit.Amount<=15000,3,
                                                                                       ifelse(d$Credit.Amount<=20000,2,1)))))))))




d$Age..years.Categorised <- ifelse(d$Age..years.<=25,1,
                                   ifelse(d$Age..years.<=39,2,
                                          ifelse(d$Age..years.<=59,3,
                                                 ifelse(d$Age..years.<=64,4,5))))


# remove the duplicate columns

colnames(d)

d2 <- d[,-c(3,6,14)]
colnames(d2)

str(d2)

# change all variables to factors
d2 <- as.data.frame(lapply(d2 , factor))
str(d2)


# find the summary of the categorical variables
# description of the split of creditable v not-creditable is given in the data description document

# Account Balance
View(as.data.frame(d %>% 
                     group_by(Creditability,Account.Balance) %>%
                     tally() %>%
                     spread(Account.Balance, n, fill = 0)))

# Payment Status of previous credit

data %>% 
  group_by(Creditability,Payment.Status.of.Previous.Credit) %>%
  tally() %>%
  spread(Payment.Status.of.Previous.Credit, n, fill = 0)

# etc...

####################
## Check Correlation
####################

# check correlation - want to move data that is highly correlated

correlationMatrix <- cor(d[,2:21])
print(correlationMatrix)

# find highly correlated variables

#highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)

# print indexes of highly correlated attributes
#print(highlyCorrelated)

res2 <- rcorr(as.matrix(d2))

str(res2)

# extract the correlation data

correlationData <- as.data.frame(res2$r)
View(correlationData)
str(correlationData)

####################
## Plot variables ##
####################

pairs(d2, col=d2$Creditability)


########################
## Feature Importance ##
########################

# run random forest model to check feature importance

View(as.data.frame(res2$r))

# check correlation
corrplot(res2$r, order = "hclust")

# a few of the parameters are highly correlated with each other so we will not use them 
# to build the model

# occupation & telephone
# type of apartment & most valuable asset
# no of credits & payment status of previous credits
# credit amount categorised & duration of credit categorised

# let's now check feature importance

############
## Random Forest
############

# creating train and test set

indexes = sample(1:nrow(d2), size=0.8*nrow(d2))
Train <- d2[indexes,] 
Test <- d2[-indexes,] 

rf50 <- randomForest(Creditability ~., data = Train, ntree=200, importance=T, proximity=T)
plot(rf50, main="")

preds <- predict(rf50, Test, type="response")

x <- as.data.frame(table(preds, Test$Creditability))
View(x)

rf50


importance(rf50)

varImpPlot(rf50)

# let's try to build a rf model with only the top 15 important variables using mean decrease accuracy & gini

newData <- d2[,c("Creditability","Account.Balance", "durationOfCreditCategorised","Purpose",
                 "Payment.Status.of.Previous.Credit","Value.Savings.Stocks",
                 "Credit.Amount.Categorised","No.of.Credits.at.this.Bank","Most.valuable.available.asset",
                 "Length.of.current.employment","Guarantors","Payment.Status.of.Previous.Credit",
                 "Instalment.per.cent")] # 79.5 accuracy

newData <- d2[,c("Creditability","Account.Balance", "durationOfCreditCategorised","Purpose",
                 "Payment.Status.of.Previous.Credit","Value.Savings.Stocks",
                 "Credit.Amount.Categorised","No.of.Credits.at.this.Bank","Most.valuable.available.asset",
                 "Length.of.current.employment","Guarantors","Payment.Status.of.Previous.Credit",
                 "Instalment.per.cent")] 


indexes = sample(1:nrow(newData), size=0.8*nrow(newData))
Train <- newData[indexes,] 
Test <- newData[-indexes,] 

rf50 <- randomForest(Creditability ~., data = Train, ntree=200, importance=T, proximity=T)
plot(rf50, main="")
preds <- predict(rf50, Test, type="response")

table(preds, Test$Creditability)

# let's try to see if fewer or more variables lead to higher accuracy

train.x = Train[,2:11]
train.y = Train[,1]

rf50 <- rfcv(train.x, train.y, cv.fold=5)
rf50

with(rf50, plot(n.var, error.cv, log="x", type="o", lwd=2))

table(preds,Test$Creditability)

# let's see with all the variables
# error is lower with 10 variables than 20 variables

indexes = sample(1:nrow(d2), size=0.8*nrow(d2))
Train <- d2[indexes,] 
Test <- d2[-indexes,] 

train.x <- Train[,2:21]
train.y <- Train[,1]

rf50 <- rfcv(train.x, train.y, cv.fold=5)

with(rf50, plot(n.var, error.cv, log="x", type="o", lwd=2))

####  tune the RF model - most accurate model has 75.67% accuracy

control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(newData)-1)
tunegrid <- expand.grid(.mtry=mtry)
rf_random <- train(Creditability~., data=newData, method="rf", metric=metric, tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random)

# run rf model with mtry = 5 - accuracy 75%

rf50 <- randomForest(Creditability ~., data = Train, mtry = 5, ntree=200, importance=T, proximity=T)
plot(rf50, main="")
preds <- predict(rf50, Test, type="response")

table(preds, Test$Creditability) 

importance(rf50)

# data$Account.Balance <- as.factor(data$Account.Balance)
# 
# data$Creditability <- as.factor(data$Creditability)
# summary(data$Creditability)

#########################
## Logistic Regression ##
#########################

# glm function


newData <- d2[,c("Creditability","Account.Balance", "Payment.Status.of.Previous.Credit","Purpose",
                 "No.of.Credits.at.this.Bank","Value.Savings.Stocks","Most.valuable.available.asset",
                 "Guarantors","Credit.Amount.Categorised",
                 "durationOfCreditCategorised","Foreign.Worker","Duration.in.Current.address",
                 "Instalment.per.cent")]


newData <- d2[,c("Creditability","Account.Balance", "Payment.Status.of.Previous.Credit","Purpose",
                 "No.of.Credits.at.this.Bank","Value.Savings.Stocks",
                 "Guarantors","Credit.Amount.Categorised",
                 "durationOfCreditCategorised","Foreign.Worker","Duration.in.Current.address",
                 "Instalment.per.cent")]

# using the 12 variables
indexes = sample(1:nrow(newData), size=0.8*nrow(newData))
Train <- newData[indexes,] 
Test <- newData[-indexes,] 

# using all variables - accuracy was 80%

set.seed(10)
indexes = sample(1:nrow(d2), size=0.8*nrow(d2))
Train <- d2[indexes,] 
Test <- d2[-indexes,] 

model <- glm(Creditability ~.,family=binomial(link='logit'),data=Train)
summary(model)

fitted.results <- predict(model,newdata=Test[,-1],type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(fitted.results != Test[,1])
print(paste('Accuracy',1-misClasificError))

model$coefficients

c <- as.data.frame(model$coefficients)
View(c)

# analyse the table of deviance

anova(model, test = "Chisq")

# plug the significant variables into the above model for a 79.5% accuracy

###################### 
## cross validation ##
######################

ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)

model <- train(Creditability ~ .,  data=Train, method="glm", family="binomial",
                 trControl = ctrl, tuneLength = 5)

model <- train(Creditability ~ .,  data=Train, method="glm", family="binomial",
               trControl = ctrl, tuneLength = 10)

pred = predict(model, newdata=Test)
confusionMatrix(data=pred, Test$Creditability)



