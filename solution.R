# Loading the packages
library(Hmisc)
library(dplyr)
library(lubridate)
library(stringr)
library(nlme)
library(randomForest)
library(xgboost)
library(ggplot2)
library(dplyr)
library(caret)
library(moments)
library(glmnet)
library(elasticnet)
library(knitr)

# Import the datasets
train <- read_csv("../input/train.csv")
test <- read_csv("../input/test.csv")
submission <- read_csv("../input/sample_submission.csv")

# combine the two datasets
train$data_flag <-"train"
test$amount_spent_per_room_night_scaled <-NA
test$data_flag <-"test"

nrow(train)
nrow(test)


summary(train$amount_spent_per_room_night_scaled)
hist((train$amount_spent_per_room_night_scaled))
# rescaled
hist(exp(train$amount_spent_per_room_night_scaled-1))

# combine the train and test
combi <- bind_rows(train,test)

# Missing Values
miss_cols=sapply(combi, function(x){sum(is.na(x))/length(x)}*100)
miss_cols

# Data Pre processing
# Missing Values imputation
combi$season_holidayed_code[is.na(combi$season_holidayed_code)] <- -1
combi$state_code_residence[is.na(combi$state_code_residence)] <- -1

# Date prcessing
combi$booking_date <- as.Date(combi$booking_date,format="%d/%m/%y")
combi$checkin_date <- as.Date(combi$checkin_date,format="%d/%m/%y")
combi$checkout_date <- as.Date(combi$checkout_date,format="%d/%m/%y")

#endcoding the unique ids
vars <- c("member_age_buckets","memberid","cluster_code","reservationstatusid_code","resort_id")
combi[,vars] <- lapply(combi[,vars],function(x){as.numeric(as.factor(x))})

# Feature Engineering
# Generate new features

# 1. Dates
# In some records, the booking date is more than checkin date.
# Inferece : Might be because of the fact that the booking date was missing and the rows were generated based on the current system date.
# Solution : We can replace such values with the checkin day, assuming those people directly approached and booked the hotel 
combi$booking_date_greater_than_checkin_flag <- ifelse(combi$booking_date>combi$checkin_date,1,0)
combi$booking_date[combi$booking_date>combi$checkin_date] <- combi$checkin_date[combi$booking_date>combi$checkin_date]

combi$booking_mnth <- month(combi$booking_date)
combi$checkin_mnth <- month(combi$checkin_date)

combi$pre_booking <- as.numeric(combi$checkin_date-combi$booking_date)
# pre booking months
combi$pre_booking <- ifelse(combi$pre_booking>=0 & combi$pre_booking<=30,1,
                            ifelse(combi$pre_booking>30 & combi$pre_booking<=60,2,
                                   ifelse(combi$pre_booking>60 & combi$pre_booking<=90,3,4)))

combi$booking_day <- as.numeric(as.factor(weekdays(combi$booking_date)))
combi$checkin_day <- as.numeric(as.factor(weekdays(combi$checkin_date)))

combi$stay_days <- as.numeric(combi$checkout_date - combi$checkin_date)

combi <- combi[combi$roomnights!=-45,]
# in some cases, we see that the roomnights is not same as the calculated stay_days. Might be extended stays or early checkouts
combi$early_checkout <- ifelse(combi$roomnights>combi$stay_days,1,0)
combi$extended_stays <- ifelse(combi$roomnights<combi$stay_days,1,0)

# 2. Members
# In some entries, total person travelling is not matching the sum of adults+children
# Inference : might be because of newborns, and they were not registered while booking
# Newly weds are more likely to go on trips
combi$newborns <- ifelse(combi$total_pax!=(combi$numberofadults+combi$numberofchildren),1,0)

# 3. check if resort and residence are in the same state
combi$same_area <- ifelse(combi$state_code_residence==combi$state_code_resort,1,0)

# remove the insignificant variables
combi$memberid <- NULL
combi$extended_stays <- NULL
# splitting the data back to train and test

train <- combi[combi$data_flag=="train",]
test <- combi[combi$data_flag=="test",]

target_var <- train$amount_spent_per_room_night_scaled
test_var <-  test$amount_spent_per_room_night_scaled

train$data_flag<-NULL
test$data_flag<-NULL
#test$amount_spent_per_room_night_scaled <-NULL

nrow(train)
nrow(test)

set.seed(1234)
RMSE = function(m, o){
  sqrt(mean((m - o)^2))
}


# xgb
# xgboost parameters
params <- list()
booster = "gblinear" 
params$objective = "reg:linear" 
params$eval_metric <- "rmse"

# Converting the data frame to matrix
xgtrain1 <-xgb.DMatrix(data=as.matrix(train[,!(colnames(train) %in% c('reservation_id','booking_date','checkin_date','checkout_date','reservationstatusid_code',"amount_spent_per_room_night_scaled"))]),label=as.matrix(target_var),missing = NA)
xgtest1 <- xgb.DMatrix(data= as.matrix(test[,!(colnames(test) %in% c('reservation_id','booking_date','checkin_date','checkout_date','reservationstatusid_code',"amount_spent_per_room_night_scaled"))]), missing = NA)

# cross-validation
#model_xgb_cv <- xgb.cv(params = params, xgtrain1, nfold=10, nrounds=1000,eta=0.01,max_depth=10,subsample=0.8,min_child_weight=12)
model_xgb_1 <- xgb.train(params = params, xgtrain1,nrounds=1000,eta=0.01,subsample=0.8,min_child_weight=4)
model_xgb_2 <- xgb.train(params = params, xgtrain1,nrounds=1500,eta=0.01,subsample=0.6)

# variable importance
#eat_imp<-data.frame(xgb.importance(feature_names=colnames(train[,!(colnames(train) %in% c("ID","Premium"))]), model=model_xgb_1))

# scoring
xgb_pred_1 <- predict(model_xgb_1, xgtest1)
xgb_pred_2 <- predict(model_xgb_2, xgtest1)


#weighted average of both predictions
xgb_pred <- 0.6*xgb_pred_1 + 0.4*xgb_pred_2
#RMSE(test_var,xgb_pred)

# xgb model
submission_xgb <- data.frame("reservation_id"=test$reservation_id,"amount_spent_per_room_night_scaled"=xgb_pred)
write_csv(submission_xgb,"submission_xgb_tuned.csv")

