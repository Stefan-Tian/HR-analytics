library(lattice)
library(caret)
library(pROC)
library(ipred)
library(e1071)
library(data.table)
library(caTools)
library(randomForest)
library(Metrics)
library(xgboost)
hr<-fread("HR_comma_sep.csv")
str(hr)
summary(hr)
hr$sales<-as.factor(hr$sales)
hr$salary<-as.factor(hr$salary)
hr$salary<-ordered(hr$salary, levels=c("low", "medium", "high"))
library(corrplot)
corrplot(cor(hr[,1:8]), method="circle")

ggplot(hr, aes(x=salary, y=satisfaction_level, fill=factor(left)))+
  geom_boxplot()+
  ggtitle("I'm not getting ENOUGH MONEY, I'm out !")+
  theme(plot.title = element_text(lineheight=2, face="bold.italic"))+
  scale_fill_manual(values=c(I("#0072B2"), I("#FF3333")), 
                    name="status",
                    breaks=c(0,1), 
                    label=c("stayed", "left"))
ggplot(hr, aes(x=salary, y=number_project, fill=factor(left)))+
  geom_boxplot()+
  ggtitle("MORE PROJECT, HIGHER SALARY?")+
  theme(plot.title=element_text(lineheight=2, face="bold.italic"))+
  scale_fill_manual(values=c(I("#0072B2"), I("#FF3333")), 
                    name="status",
                    breaks=c(0,1), 
                    label=c("stayed", "left"))

ggplot(hr, aes(factor(time_spend_company), y=average_montly_hours,
               fill=factor(left), col(factor(left))))+
  geom_boxplot(outlier.colour = NA)+
  xlab("Years_in_Company")+
  ggtitle("Spending TOO MUCH TIME, I'm out !")+
  theme(plot.title=element_text(lineheight=2, face="bold.italic"))+
  scale_fill_manual(values=c(I("#0072B2"), I("#FF3333")), 
                    name="status",
                    breaks=c(0,1), 
                    label=c("stayed", "left"))
  
ggplot(hr, aes(x=salary, y=time_spend_company, fill=factor(left)))+
  geom_boxplot(outlier.color = NA)+
  ylab("Years_in_Company")+
  ggtitle("I fill like I'm being UNDERPAID !")+
  theme(plot.title=element_text(lineheight=2, face="bold.italic"))+
  scale_fill_manual(values=c(I("#0072B2"), I("#FF3333")), 
                    name="status",
                    breaks=c(0,1), 
                    label=c("stayed", "left"))

ggplot(hr, aes(x=salary, y=average_montly_hours, fill=factor(left)))+
  geom_boxplot(outlier.color = NA)+
  ylab("average_monthly_hours")+
  ggtitle("I fill like I'm being UNDERPAID !")+
  theme(plot.title=element_text(lineheight=2, face="bold.italic"))+
  scale_fill_manual(values=c(I("#0072B2"), I("#FF3333")), 
                    name="status",
                    breaks=c(0,1), 
                    label=c("stayed", "left"))

set.seed(113)
spl<-sample.split(hr$left, SplitRatio = .7)
trainHR<-subset(hr, spl==T)
testHR<-subset(hr, spl==F)
table(trainHR$left) 
firstLog<-glm(as.factor(left)~., data=trainHR, family=binomial)
logPred1<-predict(firstLog, newdata=testHR, type="response")
logPred1
confusionMatrix(as.integer(firstPred>.5), testHR$left)
firstForest<-randomForest(as.factor(left)~., data=trainHR)
forestPred1<-predict(firstForest, newdata=testHR)
confusionMatrix(as.integer(forestPred1>0.5), testHR$left)
install.packages("Boruta")
library(Boruta)
traindata<-hr
traindata$left<-as.factor(traindata$left)
set.seed(113)
boruta_train<-Boruta(left~., data=traindata, doTrace=2)
final_boruta <- TentativeRoughFix(boruta_train)
auc(forestPred1,testHR$left)
randomForest::importance(firstForest)
TrimmedForest<-randomForest(as.factor(left)~.-promotion_last_5years-Work_accident,
                            data=trainHR)
TrimmedPred<-predict(TrimmedForest, newdata=testHR)
auc(TrimmedPred, testHR$left)
SimpleForest<-randomForest(as.factor(left)~.-promotion_last_5years-Work_accident-sales-salary,
                           data=trainHR)
SimplePred<-predict(SimpleForest, newdata=testHR)
auc(SimplePred, testHR$left)
#k-fold cross validation
k<-10
CVmodel<-train(as.factor(left)~.,
                    data=trainHR, method="cforest",
                    trControl=trainControl(method="cv", number=k, verboseIter = T))

CVmodel_trimmed<-train(as.factor(left)~.-promotion_last_5years-Work_accident-sales-salary,
               data=trainHR, method="cforest",
                trControl=trainControl(method="cv", number=k, verboseIter = T))
CV_pred<-predict(CVmodel, newdata=testHR)
CV_trimmed_pred<-predict(CVmodel_trimmed, newdata=testHR)
auc(CV_pred, testHR$left)
auc(CV_trimmed_pred, testHR$left)
