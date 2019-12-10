rm(list=ls())

library("bartMachine")
library(miscTools)
library(caret)
library(compiler)

set_bart_machine_num_cores(8)

setwd("/home/niyu/Documents/850/Project/")
data<-read.csv('data.csv')


data$Indicator<- (data$RETMONTH > 0)*1
data$Indicator[which(data$Indicator==0)]<-"NO"
data$Indicator[which(data$Indicator==1)]<-"YES"

data<-subset(data,select = -c(RETMONTH))
data<-subset(data,select = c(Close_end, retmonth_spx_end, realized_vol_spx_end, Adj_Close_end, Volume_end, fcfps_start, epsusd_start, Adj_Volume_end, sentiment_bearish_end, evebitda_start, sps_start, pb_start, grossmargin_start, payables_start, currentratio_start, sentiment_bullish_end, workingcapital_start, sbcomp_start, rnd_start, retearn_start, receivables_start, de_start, sentiment_neutral_end, pe_start, pe1_start, prefdivis_start, ncfx_start, ncfcommon_start,Indicator))

train_id<-sample(1:nrow(data), as.integer(70000), replace=FALSE)
train_data<-data[train_id,]
test_data<-data[-train_id,]


trainX <- subset(train_data,select = -c(Indicator))
trainy <- train_data$Indicator

testX <- subset(test_data,select = -c(Indicator))
testy <- test_data$Indicator

testy[which(testy=="NO")]<-0
testy[which(testy=="YES")]<-1
testy<-as.numeric(testy)

batch<-1000
iteration<-140
#m<-nrow(train_data)/batch
#id<-sample(nrow(train_data))

setwd("/home/niyu/Documents/850/Project/bartmachine")
for (i in 1:140){
  ######for every batch obeservation, build a bart
  #start<-(i-1)*batch+1
  #end<-(i*batch)
  #Number<-id[start:end]
  
  batch_id<-sample(1:nrow(train_data),batch,replace=FALSE)
  training<-train_data[batch_id,]
  
  
  y <- training$Indicator
  y<-as.factor(y)
  X <- training; X$Indicator <- NULL
  bart_machine <- bartMachine(X, y,num_trees = 50,
                              num_burn_in = 250,
                              num_iterations_after_burn_in =1000,
                              prob_rule_class = 0.5,
                              serialize = TRUE,
                              alpha = 0.95, beta = 2, k = 2, q = 0.9, nu = 3,verbose = FALSE)
  filename=paste0("bartmachine",i,'.rds')
  saveRDS(bart_machine, file = filename)
  print(100*i/iteration)
}


#### for classification in whole test set
setwd("/home/niyu/Documents/850/Project/bartmachine/")

read_bart<-function(i)
{
  filename<-paste0('bartmachine',i,'.rds')
  bartmachine<-readRDS(filename)
  return (bartmachine)
}

i_list<-seq(140)

bartlist<-lapply(i_list, read_bart)


step<-50
N<-nrow(testX)
b <- seq(step, 3000, step)
k<-lapply(seq_along(b), function(i) testX[(b-step+1)[i]:b[i], ])


#prediction function for one bartmachine
bartPredict<-function(df)
{
  result<-predict(thisbart,df)
  return (result)
}

resultdf<-matrix(0, ncol =3000, nrow = 0)


Sys.time()
for (i in 1:140){
  thisbart<-bartlist[i]
  thispredict<-unlist(lapply(k,bartPredict))
  resultdf <- insertRow(resultdf,i,thispredict)
  print(i)
}
Sys.time()


############# Generate Voting Results ############################
Indicatory<-round(resultdf)
sumy<-colSums(Indicatory)

sumy[which(sumy<=70)]<-1
sumy[which(sumy>70)]<-0
diff<-testy[1:3000]-sumy
table(diff)

