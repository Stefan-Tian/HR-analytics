library(lattice)
library(caret)
library(pROC)
library(ipred)
library(e1071)
library(data.table)
library(caTools)
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
firstLog<-glm(left~., data=trainHR, family=binomial)
firstPred<-predict(firstLog, newdata=testHR, type="response")
firstPred
table(testHR$left, firstPred>0.5)
accuracy<-function(x, y){
  (x[1,1]+x[2,2])/nrow(y)
}
accuracy(table(testHR$left, firstPred>0.5)
, testHR)
