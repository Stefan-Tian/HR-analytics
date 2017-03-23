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
setwd('~/Downloads')
hr<-fread("HR_comma_sep.csv")
str(hr)
summary(hr)
library(corrplot)
corrplot(cor(hr[,1:8]), method="circle")

#h2o can't recognize the ordered levels
hr[, `:=`(salary=as.factor(salary),left=as.factor(left), sales=as.factor(sales))]
hr[, salary:=ordered(salary,levels=c("low", "medium", "high"))]#optional

ggplot(hr, aes(x=salary, y=satisfaction_level, fill=left))+
  geom_boxplot()+
  ggtitle("I'm not getting ENOUGH MONEY, I'm out !")+
  theme(plot.title = element_text(lineheight=2, face="bold.italic"))+
  scale_fill_manual(values=c(I("#0072B2"), I("#FF3333")), 
                    name="status",
                    breaks=c(0,1), 
                    label=c("stayed", "left"))
ggplot(hr, aes(x=salary, y=number_project, fill=left))+
  geom_boxplot()+
  ggtitle("MORE PROJECT, HIGHER SALARY?")+
  theme(plot.title=element_text(lineheight=2, face="bold.italic"))+
  scale_fill_manual(values=c(I("#0072B2"), I("#FF3333")), 
                    name="status",
                    breaks=c(0,1), 
                    label=c("stayed", "left"))

ggplot(hr, aes(factor(time_spend_company), y=average_montly_hours,
               fill=left, col(left)))+
  geom_boxplot(outlier.colour = NA)+
  xlab("Years_in_Company")+
  ggtitle("Spending TOO MUCH TIME, I'm out !")+
  theme(plot.title=element_text(lineheight=2, face="bold.italic"))+
  scale_fill_manual(values=c(I("#0072B2"), I("#FF3333")), 
                    name="status",
                    breaks=c(0,1), 
                    label=c("stayed", "left"))
  
ggplot(hr, aes(x=salary, y=time_spend_company, left))+
  geom_boxplot(outlier.color = NA)+
  ylab("Years_in_Company")+
  ggtitle("I fill like I'm being UNDERPAID !")+
  theme(plot.title=element_text(lineheight=2, face="bold.italic"))+
  scale_fill_manual(values=c(I("#0072B2"), I("#FF3333")), 
                    name="status",
                    breaks=c(0,1), 
                    label=c("stayed", "left"))

ggplot(hr, aes(x=salary, y=average_montly_hours, fill=left))+
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
firstLog<-glm(left~., data=trainHR, family=binomial, maxit = 100)
logPred1<-predict(firstLog, newdata=testHR, type="response")
logPred1
confusionMatrix(as.integer(logPred1>.5), testHR$left)
firstForest<-randomForest(left~., data=trainHR)
forestPred1<-predict(firstForest, newdata=testHR)
confusionMatrix(forestPred1, testHR$left)

install.packages("Boruta")
library(Boruta)
set.seed(113)
boruta_train<-Boruta(left~., data=trainHR, doTrace=2)
final_boruta <- TentativeRoughFix(boruta_train)
auc(forestPred1,testHR$left)
randomForest::importance(firstForest)
TrimmedForest<-randomForest(left~.-promotion_last_5years-Work_accident,
                            data=trainHR)
TrimmedPred<-predict(TrimmedForest, newdata=testHR)
auc(TrimmedPred, testHR$left)
SimpleForest<-randomForest(left~.-promotion_last_5years-Work_accident-sales-salary,
                           data=trainHR)
SimplePred<-predict(SimpleForest, newdata=testHR)
auc(SimplePred, testHR$left)
#k-fold cross validation
k<-5
Control<-trainControl(method="cv", number=k, verboseIter = T)
CVmodel<-train(left~.,data=trainHR, method="cforest", trControl=Control)

CVmodel_trimmed<-train(left~.-promotion_last_5years-Work_accident,
               data=trainHR, method="cforest", trControl=Control)

CVmodel_simple<-train(left~.-promotion_last_5years-Work_accident-sales-salary,
                      data=trainHR, method="cforest", trControl=Control)
CV_pred<-predict(CVmodel, newdata=testHR)
CV_trimmed_pred<-predict(CVmodel_trimmed, newdata=testHR)
CV_simple_pred<-predict(CVmodel_simple, newdata=testHR)
auc(CV_pred, testHR$left)
auc(CV_trimmed_pred, testHR$left)
auc(CV_simple_pred, testHR$left)

localH2O <- h2o.init(nthreads = -1)
h2o.init()
train_h2o<-as.h2o(trainHR)
test_h2o<-as.h2o(testHR)
y.dep=7
x.ind=c(1:5)
DL_model<-h2o.deeplearning(x=x.ind, y=y.dep, training_frame = train_h2o, loss="CrossEntropy",
                           epochs = 50, rate=0.017, hidden=c(100, 100), activation="Rectifier",
                           seed=113)
h2o.performance(DL_model)
DL_pred<-h2o.predict(DL_model, test_h2o)
DL_pred_table<-as.data.table(DL_pred)
DL_pred_table
auc(DL_pred_table$predict, testHR$left)
library(readr)
library(stringr)
library(car)
library(Matrix)
library(xgboost)
hr<-fread("HR_comma_sep.csv") #reform the data again
trainHR<-subset(hr, spl==T)
testHR<-subset(hr, spl==F)

feature.names<-names(trainHR)[c(1:6, 8:10)]
for (f in feature.names) {
  if (class(trainHR[[f]])=="character") {
    levels <- unique(c(trainHR[[f]], testHR[[f]]))
    trainHR[[f]] <- as.integer(factor(trainHR[[f]], levels=levels))
    testHR[[f]]  <- as.integer(factor(testHR[[f]],  levels=levels))
  }
}
XG_model<-xgboost(data=data.matrix(trainHR[,feature.names, with=F]), label=trainHR$left,
                  nround=20, objective="binary:logistic", eval_metric = "auc")
testdata<-data.matrix(testHR[,feature.names, with=F])
XG_pred<-predict(XG_model, newdata=testdata)
prediction<-as.numeric(XG_pred>0.5)
error<-mean(prediction!=testHR$left)
error
print(paste("acurracy = ", 1-error))

