rm(list=ls())

library(glmnet)
library(gbm)
library(randomForest)
library(hdm)
library(rpart)
library(lsei)


#----------------------- Import Data ----------------------#
setwd("C:/Users/gyew8/OneDrive - National University of Singapore/Desktop/NUS/Y5S1/EC4308/Project")

df = read.csv("df_ml.csv")
# View(df)
# Choosing to run on resale_price
df <- subset(df, select = -ln_resale_price)

# Select the variables that end with "_dist"
dist_vars <- grep("_dist$", names(df), value = TRUE)

# Standardize the selected variables
df[dist_vars] <- scale(df[dist_vars])
df$floor_area_sqm <- scale(df$floor_area_sqm)
df$remaining_lease <- scale(df$remaining_lease)


#--------- Training, Holdout and Test set ------------- #

set.seed(2457829)

# Specify Training Size like this to keep consistency of test set with other models
gr_trsize = round(0.80*nrow(df))
# Random generate trsize numbers from 1 to nrow(df)
gr_trindicesTEMP = sample(1:nrow(df),gr_trsize)
# Training+holdout sample
gr_trainTEMP = df[gr_trindicesTEMP,]
# holdout size based on 25% of gr_trainTEMP, meaning 60-20-20 train-holdout-test
gr_holdoutsize = round(0.25*nrow(gr_trainTEMP))
# holdout indices
gr_holdoutindices = sample(1:nrow(gr_trainTEMP), gr_holdoutsize)

# holdout set
gr_holdout = gr_trainTEMP[gr_holdoutindices,]
# training set
gr_train = gr_trainTEMP[-gr_holdoutindices,]
# test set
gr_test = df[-gr_trindicesTEMP,]

x1 = model.matrix(resale_price ~ .-1, data = gr_train)
y1 = gr_train$resale_price #y for training data

x2 = model.matrix(resale_price ~ .-1, data = gr_holdout)
y2 = gr_holdout$resale_price #y for holdout data

x3 = model.matrix(resale_price ~ .-1, data = gr_test)
y3 = gr_test$resale_price #y for test data


#---------------------------GR code--------------------#
# Grid based on model building 
grid= seq(0,100,length.out=2500)


### ridge
ridge_gr.mod <- glmnet(x1, y1, alpha = 0, lambda=grid)
cv.out10 = cv.glmnet(x1, y1, alpha = 0, lambda=grid)
bestlam10r = cv.out10$lambda.min

ridge_gr.pred <- predict(ridge_gr.mod, s = bestlam10r, newx = x3)
ridge_rsme_new = sqrt(mean((ridge_gr.pred-y3)^2))
ridge_rsme_new


### lasso 
lasso_gr.mod <- glmnet(x1, y1, alpha = 1, lambda=grid)
cv.out10 = cv.glmnet(x1, y1, alpha = 1, lambda=grid)

bestlam10l = cv.out10$lambda.min #best lambda

lasso_gr.pred <- predict(lasso_gr.mod, s = bestlam10l, newx = x3)
lasso_rsme_new = sqrt(mean((lasso_gr.pred-y3)^2))
lasso_rsme_new


### elastic net
elnet_gr.mod <- glmnet(x1, y1, alpha = 0.5, lambda=grid)
cv.out10 = cv.glmnet(x1, y1, alpha = 0.5, lambda=grid)
bestlam10e = cv.out10$lambda.min

elnet_gr.pred <- predict(elnet_gr.mod, s = bestlam10e, newx = x3)
elnet_rsme_new = sqrt(mean((elnet_gr.pred-y3)^2))
elnet_rsme_new


### decision tree
big.tree = rpart(resale_price~.,method="anova",data=gr_train) 
bestcp=big.tree$cptable[which.min(big.tree$cptable[,"xerror"]),"CP"] 
best.tree = prune(big.tree,cp=bestcp) 

tree.pred = predict(best.tree, newdata=gr_test) 
tree_rsme_new = sqrt(mean((tree.pred-y3)^2))
tree_rsme_new

tree_len = 1000


### boosted tree, d=5
boost.fit = gbm(resale_price~.,data=gr_train,
                distribution='gaussian', #gaussian for squared errors
                interaction.depth=5,n.trees=tree_len,
                shrinkage=.01,cv.folds=10)

bestd5cv=gbm.perf(boost.fit, method="cv")
boost.pred = predict(boost.fit,newdata = gr_test,
                     n.trees = bestd5cv)

boostd5_rsme_new = sqrt(mean((boost.pred-y3)^2))
boostd5_rsme_new


### boosted tree, d=2
boost.fit2 = gbm(resale_price~.,data=gr_train,distribution='gaussian',
                 interaction.depth=2,n.trees=tree_len,
                 shrinkage=.01,cv.folds=10)

bestd5cv2=gbm.perf(boost.fit2, method="cv")
boost.pred2 = predict(boost.fit2,
                      newdata = gr_test,
                      n.trees = bestd5cv2)

boostd2_rsme_new = sqrt(mean((boost.pred2-y3)^2))
boostd2_rsme_new


### random forest 
forestfit <- randomForest(resale_price~.,
                          data=gr_train, ntree = 1000, mtry = 22, importance = TRUE)

rft.pred = predict(forestfit, newdata=x3)

rft_rsme_new = sqrt(mean((rft.pred-y3)^2))
rft_rsme_new


### hybrid: alternating LASSO and RF
# Fitting from LASSO
lasso_gr.pred.train = predict(lasso_gr.mod, s = bestlam10l, newx = x1)

rhat.lasso = y1 - lasso_gr.pred.train

gr_train$lasso_resid = rhat.lasso

# Random Forest on residuals of LASSO
rf.rhat.lasso = randomForest(lasso_resid~.,
                              data=subset(gr_train, select =-resale_price) , ntree = 1000, mtry = 22, importance = TRUE)

rfht.pred = predict(rf.rhat.lasso, newdata=x3)
hybridt=rfht.pred+lasso_gr.pred

hybridt_rsme_new = sqrt(mean((hybridt-y3)^2))
hybridt_rsme_new


#### Granger-Ramanathan Forecast Combinations
#In order to get weights, we need to fit all the models on the validation data first:
lasso.predc <- predict(lasso_gr.mod, s = bestlam10l, newx = x2)
ridge.predc <- predict(ridge_gr.mod, s = bestlam10r, newx = x2)
elnet.predc <- predict(elnet_gr.mod, s = bestlam10e, newx = x2)
rft.predc=predict(forestfit, newdata=x2)
boost.pred2c = predict(boost.fit2,newdata = gr_holdout,n.trees = bestd5cv2)
boost.predc = predict(boost.fit,newdata = gr_holdout,n.trees = bestd5cv)


rfht.predc = predict(rf.rhat.lasso, newdata=x2)
hybridtc=rfht.predc+lasso.predc

#-------------GR weights, no constant, constrained---------------#
#Get weights
#First, we form a design matrix where the X variables are the assorted forecasts on the validation set:
fmatu=cbind(lasso.predc,
            ridge.predc,
            elnet.predc,
            rft.predc,
            boost.pred2c,
            boost.predc,
            hybridtc)

#GR weights, no constant, all restrictions in place
gru=lsei(fmatu, y2, c=rep(1,7), d=1, e=diag(7), f=rep(0,7))
# View(gru) #Examine weights

#Combine the forecasts with nonzero weights:
combpredu=(gru[1]*lasso_gr.pred
           +gru[2]*ridge_gr.pred
           +gru[3]*elnet_gr.pred
           +gru[4]*rft.pred
           +gru[5]*boost.pred2
           +gru[6]*boost.pred
           +gru[7]*hybridt)

combpredu_rsme_new = sqrt(mean((combpredu-y3)^2))
# combpredu_rsme_new

#-------------GR weights, constant, constrained---------------#
# Redefine the X matrix for forecasts by adding a column of 
## ones - a constant regressor
fmatb=cbind(rep(1,nrow(lasso.predc)), #column of ones (for the constant)
            lasso.predc,
            ridge.predc,
            elnet.predc,
            rft.predc,
            boost.pred2c,
            boost.predc,
            hybridtc) 

temp=diag(8)
temp[1,1]=0

#Find the GR weights under constraints, bu with constant in the regression:
grb=lsei(fmatb, y2, c=c(0,rep(1,7)), d=1, e=temp, f=rep(0,8))
# View(grb)
#From the forecasts using nonzero weights:
combpredb=(grb[1]
           +grb[2]*lasso_gr.pred
           +grb[3]*ridge_gr.pred
           +grb[4]*elnet_gr.pred
           +grb[5]*rft.pred
           +grb[6]*boost.pred2
           +grb[7]*boost.pred
           +grb[8]*hybridt)

combpredb_rsme_new = sqrt(mean((combpredb-y3)^2))
# combpredb_rsme_new

#---------------------GR, no constant, no constrain----------------#
#unrestricted weights: no constraints, no constant
grunr=lsei(fmatu, y2)

#Form combined forecast (almost all weights nonzero, 
## so do vector product to sum):
combpredur=cbind(lasso_gr.pred,
                 ridge_gr.pred,
                 elnet_gr.pred,
                 rft.pred,
                 boost.pred2,
                 boost.pred,
                 hybridt)%*%grunr 

combpredur_rsme_new = sqrt(mean((combpredur-y3)^2))
# combpredur_rsme_new

#--------------------GR, constant, no constrain---------------------#
# unrestricted weights: no constraints, but include constant
grunrc=lsei(fmatb, y2)

combpredurc=cbind(rep(1,nrow(lasso_gr.pred)),
                  lasso_gr.pred,
                  ridge_gr.pred,
                  elnet_gr.pred,
                  rft.pred,
                  boost.pred2,
                  boost.pred,
                  hybridt)%*%grunrc 

combpredurc_rsme_new = sqrt(mean((combpredurc-y3)^2))
# combpredurc_rsme_new

#------------------Put GR stuff into a (nice?) table-------------------#
# first include NA for no constant GRs
gru <- c(NA, gru)
grunr <- c(NA, grunr)

# include RMSE as last value
gru <- c(gru, combpredu_rsme_new)
grb <- c(grb, combpredb_rsme_new)
grunr <- c(grunr, combpredur_rsme_new)
grunrc <- c(grunrc, combpredurc_rsme_new)

# individual methods used
methods <- c("constant", 
             "LASSO", 
             "Ridge", 
             "Elastic Net", 
             "Random Forest", 
             "Boosted Trees (d=2)",
             "Boosted Trees (d=5)", 
             "Hybrid (LASSO + RF)",
             "RMSE")

GR_weights_rmse <- data.frame(Methods = methods, 
                              GR.nconstant.constrain = gru, 
                              GR.constant.constrain = grb, 
                              GR.nconstant.nconstrain = grunr, 
                              GR.constant.nconstrain = grunrc)