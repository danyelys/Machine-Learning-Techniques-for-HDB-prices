rm(list=ls())
#Import Libraries
library(ISLR)
library(pls)
library(glmnet)
library(hdm)
library(rpart)
library(randomForest)
library(gbm)
library(ggplot2)
library(reshape2)

#----------------------- Import Data ----------------------#
setwd("C:/Users/gyew8/OneDrive - National University of Singapore/Desktop/NUS/Y5S1/EC4308/Project")

df = read.csv("df_ml.csv")
#View(df)

# Choosing to run on resale_price
df <- subset(df, select = -ln_resale_price)

# Select the variables that end with "_dist"
dist_vars <- grep("_dist$", names(df), value = TRUE)

# Standardize the selected variables
df[dist_vars] <- scale(df[dist_vars])
df$floor_area_sqm <- scale(df$floor_area_sqm)
df$remaining_lease <- scale(df$remaining_lease)

#----------------------- Train-Test Split ----------------------#

set.seed(2457829)

# Specify Training Size (maybe use perc of sample size)
trsize = round(0.80*nrow(df))

### Create some matrices for prediction purposes ###
X = model.matrix(resale_price ~., data = df)
#View(X)
X=X[, colnames(X)!= "(Intercept)"] 
Y = df$resale_price #define Y

### train/test split ###
trindices = sample(1:nrow(X),trsize)  # randomly generate trsize numbers from 1 to the nrow(df)
train = df[trindices,]   # get the training sample (getting only rows whose indices are in trindices)
test = df[-trindices,]   # get the testing sample (remove rows whose indices are in trindices)
Ytrain = Y[trindices]
Ytest = Y[-trindices]

#-------------- OLS ---------------#
ols <- lm(resale_price~., data = train)

ols_pred <- predict(ols, newdata=test)

ols_rmse <- sqrt(mean((test$resale_price - ols_pred)^2))

#-------------- Ridge 10 fold CV ---------------#
#user-defined grid
grid = seq(0,1000,length.out=1000)
grid2 = seq(0,40,length.out=1000)

ridge.mod <- glmnet(X[trindices,], Y[trindices], alpha = 0, 
                    lambda = grid, thresh = 1e-12)

ridge_cv = cv.glmnet(X[trindices,], Y[trindices], alpha = 0, lambda=grid)
plot(ridge_cv$lambda,ridge_cv$cvm, #  manually pull out lambdas and MSE 
     ## computed in this CV algorithm
     main="10-fold CV Ridge", xlab="Lambda", ylab="CV MSE")

ridge_minlambda = ridge_cv$lambda.min #This lambda returns the smallest CV MSE
ridge_pred <- predict(ridge.mod, s = ridge_minlambda, newx = X[-trindices,])

ridge_rmse = sqrt(mean((ridge_pred-Y[-trindices])^2))
ridge_rmse
ridge_minlambda

ridge2.mod <- glmnet(X[trindices,], Y[trindices], alpha = 0, 
                     lambda = grid2, thresh = 1e-7)

ridge2_cv = cv.glmnet(X[trindices,], Y[trindices], alpha = 0, lambda=grid2)
plot(ridge2_cv$lambda,ridge2_cv$cvm, #  manually pull out lambdas and MSE 
     ## computed in this CV algorithm
     main="10-fold CV Ridge", xlab="Lambda", ylab="CV MSE")

ridge2_minlambda = ridge2_cv$lambda.min #This lambda returns the smallest CV MSE
ridge2_pred <- predict(ridge2.mod, s = ridge2_minlambda, newx = X[-trindices,])

ridge2_rmse = sqrt(mean((ridge2_pred-Y[-trindices])^2))
ridge2_rmse
ridge2_minlambda


#-------------- LASSO 10 fold CV ---------------#
lasso_cv = cv.glmnet(X[trindices,], Y[trindices], alpha = 1, lambda = grid, thresh = 1e-7)
plot(lasso_cv$lambda,lasso_cv$cvm, #  manually pull out lambdas and MSE 
     ## computed in this CV algorithm
     main="10-fold CV LASSO", xlab="Lambda", ylab="CV MSE")

minlasso_lambda = lasso_cv$lambda.min

lasso_pred <- predict(lasso_cv, s = minlasso_lambda, newx = X[-trindices,])
lasso_rmse = sqrt(mean((lasso_pred-Y[-trindices])^2))
lasso_rmse
minlasso_lambda


grid3 = seq(0,100,length.out=2500)
lasso2_cv = cv.glmnet(X[trindices,], Y[trindices], alpha = 1, lambda = grid3, thresh = 1e-7)
plot(lasso2_cv$lambda,lasso2_cv$cvm, #  manually pull out lambdas and MSE 
     ## computed in this CV algorithm
     main="10-fold CV LASSO", xlab="Lambda", ylab="CV MSE")

minlasso2_lambda = lasso2_cv$lambda.min

lasso2_pred <- predict(lasso2_cv, s = minlasso2_lambda, newx = X[-trindices,])
lasso2_rmse = sqrt(mean((lasso2_pred-Y[-trindices])^2))
lasso2_rmse
minlasso2_lambda


#----------- Elastic Net 10 fold CV ------------#
al=0.5 #alpha value
elasnet <- glmnet(X[trindices,], Y[trindices], alpha = al)
elasnet_cv  = cv.glmnet(X[trindices,], Y[trindices], alpha = al)
plot(elasnet_cv$lambda,elasnet_cv$cvm, main="10-fold CV Elastic Net", 
     xlab="Lambda", ylab="CV MSE")

minelasnet_lambda = elasnet_cv$lambda.min
elasnet_pred <- predict(elasnet_cv,s = minelasnet_lambda,newx = X[-trindices,])
elasnet_rmse = sqrt(mean((elasnet_pred - Y[-trindices])^2))
elasnet_rmse
minelasnet_lambda


elasnet2 <- glmnet(X[trindices,], Y[trindices], alpha = al, lambda = grid3)
elasnet2_cv  = cv.glmnet(X[trindices,], Y[trindices], alpha = al, lambda=grid3)
plot(elasnet2_cv$lambda,elasnet2_cv$cvm, main="10-fold CV Elastic Net", 
     xlab="Lambda", ylab="CV MSE")

minelasnet2_lambda = elasnet2_cv$lambda.min
elasnet2_pred <- predict(elasnet2_cv,s = minelasnet2_lambda,newx = X[-trindices,])
elasnet2_rmse = sqrt(mean((elasnet2_pred - Y[-trindices])^2))
elasnet2_rmse
minelasnet2_lambda


#-------------PCR 10-fold CV--------------#
pcr_mod_10fold=pcr(resale_price~., data=df, subset=trindices, 
                   scale=TRUE, validation="CV")

summary(pcr_mod_10fold)

#validation plot is from the package pls which helps to plot the result
validationplot(pcr_mod_10fold, val.type="MSEP", main="10-fold CV",
               legendpos = "topright")

ncomprdm_pcr_10fold = selectNcomp(pcr_mod_10fold, method = "randomization", plot = TRUE, main="Randomized Selection")

#Predict using the desired component number and display MSE
pcr_preds_10fold_rdm = predict(pcr_mod_10fold, newdata=X[-trindices,], ncomp=ncomprdm_pcr_10fold)

#OOS MSE
pcr_rmse_10fold_rdm = sqrt(mean((test$resale_price - pcr_preds_10fold_rdm)^2))
pcr_rmse_10fold_rdm

######################----PCR GRAPHS----######################
### Plotting the first 2 PC of PCR and coloured by resale price
# Get the scores of the first two principal components
scores <- pcr_mod_10fold$scores[,1:2]

# Convert the scores to a data frame
scores_df <- as.data.frame(scores)
colnames(scores_df) <- c("PC1", "PC2")

# Add the response variable to the data frame
scores_df$resale_price <- train$resale_price

# Create the plot
ggplot(scores_df, aes(x = PC1, y = PC2, color = resale_price)) +
  geom_point() +
  scale_color_gradient(low = "blue", high = "red") +
  labs(x = "First Principal Component", y = "Second Principal Component", color = "resale_price")


####### Examining the first 2 Principal Components
### FIRST PC
loadings <- pcr_mod_10fold$loadings[,1]

# Convert the loadings to a data frame
loadings_df <- as.data.frame(loadings)
loadings_df$Variable <- rownames(loadings_df)
colnames(loadings_df) <- c("Loading", "Variable")

# Create a new column for the sign of the loading
loadings_df$Sign <- ifelse(loadings_df$Loading < 0, "Negative", "Positive")

# Take the absolute value of the loadings
loadings_df$Loading <- abs(loadings_df$Loading)

# Order the data frame by the magnitude of the loadings
loadings_df <- loadings_df[order(loadings_df$Loading, decreasing = TRUE),]

# Create the plot
ggplot(loadings_df, aes(x = reorder(Variable, Loading), y = Loading, fill = Sign)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("Positive" = "green", "Negative" = "red")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Variable", y = "Loading", fill = "Sign")


### SECOND PC
loadings <- pcr_mod_10fold$loadings[,2]

# Convert the loadings to a data frame
loadings_df <- as.data.frame(loadings)
loadings_df$Variable <- rownames(loadings_df)
colnames(loadings_df) <- c("Loading", "Variable")

# Create a new column for the sign of the loading
loadings_df$Sign <- ifelse(loadings_df$Loading < 0, "Negative", "Positive")

# Take the absolute value of the loadings
loadings_df$Loading <- abs(loadings_df$Loading)

# Order the data frame by the magnitude of the loadings
loadings_df <- loadings_df[order(loadings_df$Loading, decreasing = TRUE),]

# Create the plot
ggplot(loadings_df, aes(x = reorder(Variable, Loading), y = Loading, fill = Sign)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("Positive" = "green", "Negative" = "red")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Variable", y = "Loading", fill = "Sign")


#-------------PLS 10 fold CV--------------#
pls_mod_10fold=plsr(resale_price~., data=df, subset=trindices, 
                    scale=TRUE, validation="CV")

#validation plot is from the package pls which helps to plot the result
validationplot(pls_mod_10fold, val.type="MSEP", main="10-fold CV",
               legendpos = "topright")

ncomprdm_pls_10fold = selectNcomp(pls_mod_10fold, method = "randomization", plot = TRUE, main="Randomized Selection")

#Predict using the desired component number and display MSE:
pls_preds_10fold_rdm = predict(pls_mod_10fold, newdata=test, ncomp=ncomprdm_pls_10fold)

#OOS MSE
pls_rmse_10fold_rdm = sqrt(mean((test$resale_price - pls_preds_10fold_rdm)^2))
pls_rmse_10fold_rdm

######################----PLS GRAPHS----######################
### Plotting first 2 components of PLS and coloured by resale price
# Get the scores of the first two components
scores <- pls_mod_10fold$scores[,1:2]

# Convert the scores to a data frame
scores_df <- as.data.frame(scores)
colnames(scores_df) <- c("Comp1", "Comp2")

# Add the response variable to the data frame
scores_df$resale_price <- train$resale_price

# Create the plot
ggplot(scores_df, aes(x = Comp1, y = Comp2, color = resale_price)) +
  geom_point(alpha=0.5) +
  scale_color_gradient(low = "blue", high = "red") +
  labs(x = "First Component", y = "Second Component", color = "Resale Price")


####### Examining the first and second component in PLS
### FIRST COMPONENT
loadings <- pls_mod_10fold$loadings[,1]

# Convert the loadings to a data frame
loadings_df <- as.data.frame(loadings)
loadings_df$Variable <- rownames(loadings_df)
colnames(loadings_df) <- c("Loading", "Variable")

# Create a new column for the sign of the loading
loadings_df$Sign <- ifelse(loadings_df$Loading < 0, "Negative", "Positive")

# Take the absolute value of the loadings
loadings_df$Loading <- abs(loadings_df$Loading)

# Order the data frame by the magnitude of the loadings
loadings_df <- loadings_df[order(loadings_df$Loading, decreasing = TRUE),]

# Create the plot
ggplot(loadings_df, aes(x = reorder(Variable, Loading), y = Loading, fill = Sign)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("Positive" = "green", "Negative" = "red")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Variable", y = "Loading", fill = "Sign")

### SECOND COMPONENT
loadings <- pls_mod_10fold$loadings[,2]

# Convert the loadings to a data frame
loadings_df <- as.data.frame(loadings)
loadings_df$Variable <- rownames(loadings_df)
colnames(loadings_df) <- c("Loading", "Variable")

# Create a new column for the sign of the loading
loadings_df$Sign <- ifelse(loadings_df$Loading < 0, "Negative", "Positive")

# Take the absolute value of the loadings
loadings_df$Loading <- abs(loadings_df$Loading)

# Order the data frame by the magnitude of the loadings
loadings_df <- loadings_df[order(loadings_df$Loading, decreasing = TRUE),]

# Create the plot
ggplot(loadings_df, aes(x = reorder(Variable, Loading), y = Loading, fill = Sign)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("Positive" = "green", "Negative" = "red")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Variable", y = "Loading", fill = "Sign")


#-------------PCR-LASSO--------------#
# principal component matrix
pc_mat = pcr_mod_10fold$scores

# user defined grid to find min lambda
newgrid = seq(0,100,length.out=200)

# performing LASSO CV10
pcr_lasso_cv <- cv.glmnet(pc_mat, Y[trindices], alpha = 1, lambda = newgrid)


### Plotting the Graph to Visualise MSE and lambda ###
# Plot MSE vs log(lambda)
plot(pcr_lasso_cv$lambda, pcr_lasso_cv$cvm, type='l', xlim = range(0,100), xlab="lambda", ylab="MSE", main="MSE vs lambda for LASSO")
# Add a point for the minimum MSE
points(pcr_lasso_cv$lambda.min, min(pcr_lasso_cv$cvm), col="red", pch=19)


# Generating test matrix of principal component
pc_mat_test <- predict(pcr_mod_10fold, X[-trindices,], type = "scores")

# calculating the RMSE of PCR-LASSO
pcr_lasso_pred <- predict(pcr_lasso_cv, s = pcr_lasso_cv$lambda.min , newx = pc_mat_test)
pcr_lasso_rmse = sqrt(mean((pcr_lasso_pred-Y[-trindices])^2))
pcr_lasso_rmse

######################----PCR-LASSO GRAPHS----######################
### GRAPHING log of coefficients against PC 
# Get the coefficients at lambda.min
pcrLASSO_lambda.min <- pcr_lasso_cv$lambda.min
pcrLASSSO_coefficients <- as.vector(coef(pcr_lasso_cv, s = pcrLASSO_lambda.min))[-1] 

# Convert the coefficients to a data frame
pcrLASSO_coefficients_df <- as.data.frame(pcrLASSSO_coefficients)
pcrLASSO_coefficients_df$Variable <- rownames(pcrLASSO_coefficients_df)
colnames(pcrLASSO_coefficients_df) <- c("Coefficient", "Variable")

# Filter out the variables with zero coefficient
pcrLASSO_coefficients_df <- pcrLASSO_coefficients_df[pcrLASSO_coefficients_df$Coefficient != 0,]

# Create a new column for the sign of the coefficient
pcrLASSO_coefficients_df$Sign <- ifelse(pcrLASSO_coefficients_df$Coefficient < 0, "Negative", "Positive")

# Take the natural logarithm of the absolute value of the coefficients
pcrLASSO_coefficients_df$Coefficient <- log(abs(pcrLASSO_coefficients_df$Coefficient))

# Create the plot of log coefficient against components
ggplot(pcrLASSO_coefficients_df, aes(x = reorder(Variable, Coefficient), y = Coefficient, fill = Sign)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("Positive" = "green", "Negative" = "red")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Variable", y = "Log of Coefficient", fill = "Sign")


### Looking at the weights assigned to the top 3 PC (62,63,64) from before 
# Get the loadings of the principal components
loadings <- pcr_mod_10fold$loadings[,62:64]

# Convert the loadings to a data frame and reshape for plotting
loadings_df <- as.data.frame(loadings)
loadings_df$Variable <- rownames(loadings_df)
loadings_melt <- melt(loadings_df, id.vars = "Variable", variable.name = "Component", value.name = "Loading")

# Create the plot for the loadings of each PC
ggplot(loadings_melt, aes(x = Variable, y = Loading, fill = Component)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 70, hjust = 1)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(x = "Variable", y = "Loading", fill = "Component")


#--------------- Decision Trees ----------------#
temp_bigtree = rpart(resale_price~.,
                     data=train, method = "anova", minsplit = 10, 
                     cp = 0.00001, maxdepth = 30)

options(repr.plot.width=7, repr.plot.height=7) 
#We now examine the CV plot, 
## invoked by plotcp() function on the fitted tree:
plotcp(temp_bigtree) #CV plot

#Extract minimum error from cv
minxerror_pos = which.min(temp_bigtree$cptable[,"xerror"])
#Extract cv that returns the minimum error
bestcp=temp_bigtree$cptable[minxerror_pos,"CP"]
#Prune tree with respect to the best cp
bestAnova =  prune(temp_bigtree,cp = bestcp)

#Predict on test data with final model
Xtest = subset(test, select = -c(resale_price))
yAnova = predict(bestAnova,newdata = Xtest)
decisiontree_rsme = sqrt(mean((yAnova - Y[-trindices])^2))
decisiontree_rsme


#Variable Importance
variable_impt <- as.data.frame(bestAnova$variable.importance)
variable_impt <- head(variable_impt, 10)

#Settings for graph
par(las =1)
par(mar=c(4,8,3,3))
barplot(height =head(bestAnova$variable.importance,10), names = rownames(variable_impt), horiz = T, cex.names = 0.6, col = "darkblue")


#---------------- Random Forest ----------------#
#Forestfit with large ntree and no variable importance 
rffit2 <- randomForest(resale_price~.,data=train,ntree=5000,maxnodes=40, mtry=21)
#CV plot of OOB-Error  

### Set up a two-panel plot layout to plot the mse vs ntree ###
par(mfrow=c(1,2))
# Get the OOB error for the largest ntree
max_ntree_oob_error <- tail(rffit2$mse, n=1)
# Calculate the ±1% threshold
lower_threshold <- 0.99 * max_ntree_oob_error
upper_threshold <- 1.01 * max_ntree_oob_error

# Zoomed Plot
# Subset the OOB error data starting from ntree = 200
oob_error <- rffit2$mse[200:length(rffit2$mse)]
# Create a new x-axis for the plot
x_axis <- seq(200, length(rffit2$mse), by=1)
# Plot the OOB error
plot(x_axis, oob_error, type="l", xlab="Number of Trees", ylab="OOB Error")
# Add horizontal lines at the ±1% thresholds
abline(h=lower_threshold, col="blue", lty=2, lwd=2)
abline(h=upper_threshold, col="blue", lty=2, lwd=2)

# Everything Plot
# Subset the OOB error data starting from ntree = 1
oob_error <- rffit2$mse[1:length(rffit2$mse)]
# Create a new x-axis for the plot
x_axis <- seq(1, length(rffit2$mse), by=1)
# Plot the OOB error
plot(x_axis, oob_error, type="l", xlab="Number of Trees", ylab="OOB Error")


### Forestfit with variable importance and ntree=1000 
forestfit <- randomForest(resale_price~.,
                          data=train, ntree = 1000, mtry = 22, importance = TRUE)

#Predict on test data with final model
yforest = predict(forestfit,newdata = Xtest)
forest_rsme = sqrt(mean((yforest - Y[-trindices])^2))
forest_rsme

#Variable Importance for random Forest with 1000 trees
importance(forestfit)
varImpPlot(forestfit, n.var = 8, sort = TRUE)


#----------- Boosted Trees - depth 2 ------------#
#Train model with 10000 trees with 10-fold CV
boostfit = gbm(resale_price~.,
               data=train,
               distribution='gaussian',
               bag.fraction = .5,
               interaction.depth=2,
               n.trees=10000,shrinkage=.01, cv.folds = 10)
#Optimal Iteractions from 10-fold CV for d = 2
bestd2cv=gbm.perf(boostfit, method="cv") 

#Predict on test data with final model
yboost = predict(boostfit, newdata = Xtest, n.trees = bestd2cv)
boostedtree_rsme = sqrt(mean((yboost - Y[-trindices])^2))
boostedtree_rsme


#----------- Boosted Trees - depth 5 ------------#
boostfit2 = gbm(resale_price~.,
                data=train,
                distribution='gaussian',
                bag.fraction = .5,
                interaction.depth=5,
                n.trees=10000,shrinkage=.01, cv.folds = 10)
#Optimal Iteractions from 10-fold CV for d = 5
bestd5cv=gbm.perf(boostfit2, method="cv") 

#Predict on test data with final model
yboost2 = predict(boostfit2, newdata = Xtest, n.trees = bestd5cv)
boostedtree2_rsme = sqrt(mean((yboost2 - Y[-trindices])^2))
boostedtree2_rsme

#Variable Importance for d = 5
par(mar = c(5, 8, 1, 1))
summary.gbm(boostfit, cBars = 10,las = 2)


#-----------------Hybrid (LASSO-RF)-----------------#
lasso2_train_pred <- predict(lasso_cv, s = minlasso2_lambda, newx = X[trindices,])

rhat.lasso = Ytrain - lasso2_train_pred

train$lasso_resid = rhat.lasso

rf.rhat.lasso = randomForest(lasso_resid~.,
                             data=subset(train, select =-resale_price) , ntree = 1000, mtry = 22, importance = TRUE)

rfht.pred = predict(rf.rhat.lasso, newdata=test)
hybridt=rfht.pred+lasso2_pred

hybridt_rsme = sqrt(mean((hybridt-Ytest)^2))
hybridt_rsme

all_rmse = c(ridge2_rmse, lasso2_rmse, elasnet2_rmse, 
             pcr_rmse_10fold_rdm, pls_rmse_10fold_rdm, 
             pcr_lasso_rmse, decisiontree_rsme, forest_rsme, 
             boostedtree_rsme, boostedtree2_rsme, hybridt_rsme)
