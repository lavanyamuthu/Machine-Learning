library(forecast)
library(xts)
path <- "C:\\Users\\zkdzgop\\Desktop\\Algorithm\\Datatrend"
setwd(path)
#sampledata <- read.csv("CSVFile_2016-08-29T12_55_43.csv",header = TRUE,stringsAsFactors = FALSE)
sampledata <- read.csv("CSVFile_Count.csv",header = TRUE,stringsAsFactors = FALSE)
str(sampledata)
library(quantmod)
dt <- sample(nrow(sampledata),nrow(sampledata)*(0.8))
sampledata[dt,]
sampledata[-dt,]
train <- sampledata[dt,]
test <- sampledata[-dt,]
testinp <- data.frame(subset(test,select=-cnt))
count.lm <- lm(cnt ~ Duration.since, data=train)
testop <- predict(count.lm,testinp,type="response")
testop
actualop <- test$cnt
actuals_preds <- data.frame(cbind(actuals=test$cnt, predicteds=testop))
correlation_accuracy <- cor(actuals_preds)
correlation_accuracy #98.7% 98.56% 95.38% 97.89% 98.27% 98.35% 98.13%
actuals_preds
 
rmse <- sqrt(mean((test$cnt - testop)^2))
rmse

#ARIMA models
 
#ACF
tsdata <- read.csv("CSVFile_Count - for TS.csv",header = TRUE,stringsAsFactors = FALSE)
plot(tsdata)
library(ggplot2)
ggplot(tsdata,aes(Month,cnt))+geom_point()
ggplot(tsdata,aes(DurationSince,cnt))+geom_point()
str(tsdata)
inputts <- tsdata[1:15,]$cnt
inputpred <- tsdata[16:21,]$cnt
myts <- ts(inputts,start=c(2014,11),frequency=12)
plot(myts)
components.myts <- decompose(myts)
fit <- stl(myts,s.window = "period")
acf(myts,12) #AR
pacf(myts,12) #not MA
#PACF
mytsstationary <- diff(myts,differences = 2)
plot(mytsstationary)
acf(mytsstationary,12)
pacf(mytsstationary,12)
 
fitArima1 <- arima(myts,order=c(1,1,0))
install.packages("lmtest")
library(lmtest)
coeftest(fitArima1)
acf(fitArima1$residuals)
pacf(fitArima1$residuals)
predict(fitArima1,n.ahead=6)
 
#model 2
fitArima2 <- arima(myts,order=c(1,1,1))
#install.packages("lmtest")
library(lmtest)
coeftest(fitArima2)
acf(fitArima2$residuals)
pacf(fitArima2$residuals)
predict(fitArima2,n.ahead=6)
 
#model 3
fitArima3 <- arima(myts,order=c(0,1,1))
library(lmtest)
coeftest(fitArima3)
acf(fitArima3$residuals)
pacf(fitArima3$residuals)
predict(fitArima3,n.ahead=6)
 
#Outliers check in count
library(ggplot2)
qplot(data = sampledata, x = sampledata$cnt) + ylab("Count")
ggplot(sampledata,aes(Duration.since,cnt))+geom_point()
ggplot(sampledata,aes(Duration.since,cnt))+geom_point(aes(color = Month))
ggplot(sampledata,aes(Duration.since,cnt))+geom_point(aes(shape = factor(Year)))
 
install.packages("reshape2")
library(reshape2)
ggplot(sampledata,aes(x=Duration.since,y=cnt))+geom_boxplot()+facet_grid(.~Year)
 
require(forecast)
selectsample <- subset(sampledata,select=c(Duration.since,cnt))
selectsample1 <- log10(selectsample)
str(selectsample1)
ggplot(selectsample1,aes(Duration.since,cnt))+geom_point()
ARIMAfit <- auto.arima(log10(selecttrain), approximation=FALSE,trace=FALSE)
summary(ARIMAfit)
 
class(selectsample)
plot(diff(as.double(selectsample)),ylab="Differenced Tractor Sales")
 
cntsampletrain <- data.frame(tsdata[1:16,])
cntsampletest <- data.frame(tsdata[17:21,])
 
cnttimeseries <- ts(cntsampletrain, frequency=12, start=c(2014,1))
cnttimeseries
plot.ts(cnttimeseries)
#balmyts <- ts(balsampledata,start=c(2014,11),end=c(2016,7),frequency=12)
options("scipen"=100,"digits"=4)
cntforecst <- HoltWinters(cnttimeseries,gamma=FALSE)
cntforecst
cntforecst$SSE
cntforecst2 <- forecast.HoltWinters(cntforecst,h=5)
cntforecst2
colnames(cntsampletrain) <- "cnt"
colnames(cntsampletest) <- "cnt"
colnames(tsdata)
actualcnt <- cntsampletest$cnt
class(actualcnt)
cntforecst2
#predcnt <- c(24307508,24369873,24491808,24725033,24905074)
predcnt <- c(24416338,24659168,24901999,25144829,25387659,24416338)
class(predcnt)
 
actuals_cnt_preds <- data.frame(cbind(actuals=cntsampletest$cnt, predicteds=predcnt))
correlation_accuracy_cnt <- cor(actuals_cnt_preds)
correlation_accuracy_cnt # 97.9%
 
str(tsdata)
acf(tsdata)
pacf(tsdata)
 
arimadata <- read.csv("CSVFile_Count - for TS2.csv",header = TRUE,stringsAsFactors = FALSE)
plot(arimadata)
library(ggplot2)
ggplot(arimadata,aes(Month,cnt))+geom_point()
ggplot(arimadata,aes(DurationSince,cnt))+geom_point()
str(arimadata)
inputts <- arimadata[1:15,]$cnt
inputpred <- arimadata[16:21,]$cnt
myts <- ts(inputts,start=c(2014,11),frequency=12)
#fit <- stl(inputts,t.window = 12,s.window = "periodic",robust=TRUE)
#fit <- decompose(inputts,type="multiplicative")
plot(myts)
acf(myts)
pacf(myts)
plot(diff(log(myts),3))
components.myts <- decompose(myts)
fit <- stl(myts,s.window = "period")
acf(diff(myts,3),12) #AR
pacf(diff(myts,3),12) #not MA
#PACF
mytsstationary <- diff(myts)
plot(mytsstationary)
acf(mytsstationary,12)
pacf(mytsstationary,12)