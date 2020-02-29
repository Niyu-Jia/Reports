library(MASS)
library(KernSmooth)
library(locpol)

data<-read.csv('/home/niyu/Documents/Project/Point72/data.csv')

#x<-scale(data$Signal,center=TRUE,scale=TRUE)

#y<-scale(data$ClosePrice,center=TRUE,scale=TRUE)
x<-data$Signal
y<-data$ClosePrice

sigma_x<-sd(data$Signal)


bw2=dpill(y,x)
fit<-locPolSmootherC(x,y,xeval=x,bw2,deg=1,EpaK)
mu1<-fit$beta0
beta<-fit$beta1


library(ggplot2)
ggplot(data.frame(x,mu1,y),aes(x=x,y=y))+geom_point(color='red',shape='*',size=5)+geom_line(aes(x,mu1))

#plot(x,mu1,col='red')
#points(x,y)


bw3=dpill((y-mu1)^2,x)
fit2<-locPolSmootherC(x,(y-mu1)^2,xeval=x,bw3,deg=1,EpaK)
mu2<-fit2$beta0

ggplot(data.frame(x,mu2,y),aes(x=x,y=(y-mu1)^2))+geom_point(color='red',shape='*',size=5)+geom_line(aes(x,mu2))


#rho<-sigma_x*beta/(sigma_x^2*beta^2+mu2-mu1^2)^(1/2)
rho<-sigma_x*beta/(sigma_x^2*beta^2+mu2)^(1/2)

ggplot(data.frame(x,rho),aes(x=x,y=rho))+geom_point(color='black',shape='*',size=5)