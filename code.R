library(tidyverse)
library(tidyr)
library(corrplot)
library(mice)
library(rpart)
library(randomForest)
library(xgboost)
library(e1071)
library(nnet)
library(forcats)
library(VIM)
library(caret)
library(pROC)


# DATA LOADING AND EXPLORATION
initial_data <- read.csv("LUBS5990M_courseworkData_202425.csv", stringsAsFactors = FALSE, encoding='UTF-8')
dim(initial_data)
str(initial_data)
colnames(initial_data)
head(initial_data)
summary(initial_data)
table(initial_data$success)


# DATA CLEANING AND PREPARATION

## calculating sold tokens
initial_data$distributed_in_ico <- as.numeric(initial_data$distributed_in_ico)
initial_data$sold_tokens[is.na(initial_data$sold_tokens) & 
                           !is.na(initial_data$token_for_sale) & 
                           !is.na(initial_data$distributed_in_ico)] <- 
  initial_data$token_for_sale[is.na(initial_data$sold_tokens) & 
                                !is.na(initial_data$token_for_sale) & 
                                !is.na(initial_data$distributed_in_ico)] * 
  (initial_data$distributed_in_ico[is.na(initial_data$sold_tokens) & 
                                     !is.na(initial_data$token_for_sale) & 
                                     !is.na(initial_data$distributed_in_ico)] / 100)

## encoding categorical variables
initial_data$success <- ifelse(initial_data$success == 'Y', 1, 0)
binary_columns <- c("whitelist", "kyc", "bonus", "mvp", "ERC20")
initial_data[binary_columns] <- lapply(initial_data[binary_columns], function(x) as.numeric(x == "Yes" | x == 1))

## handling dates
initial_data$ico_start <- as.Date(initial_data$ico_start, format="%d/%m/%Y")
initial_data$ico_end <- as.Date(initial_data$ico_end, format="%d/%m/%Y")

## feature engineering
### calculating ico duration
initial_data$ico_duration <- as.numeric(initial_data$ico_end - initial_data$ico_start)

## handling missing data
data <- initial_data[!is.na(initial_data$ico_start) & !is.na(initial_data$ico_end), ]
names(data)[1] <- "country" 
### checking missing data values
missing_initial_data <- colSums(is.na(initial_data))
missing_initial_data
### deleting rows containing missing values
delete_rows <- c("price_usd", "country", "kyc", "bonus", "ico_duration")
data <- data %>% drop_na(all_of(delete_rows))
### deleting non-critical columns with high missing values
delete_column <- c("linkedin_link", "github_link", "website", 
                   "link_white_paper", "pre_ico_start", "pre_ico_end")
data <- data %>% dplyr::select(-all_of(delete_column))
### creating binary features
data$has_whitelist <- ifelse(is.na(data$whitelist), 0, 1)
data$whitelist <- NULL
data$has_min_investment <- ifelse(is.na(data$min_investment), 0, 1)
data$min_investment <- NULL
data$has_mvp <- ifelse(is.na(data$mvp), 0, 1)
data$mvp <- NULL
data$has_restricted_areas <- ifelse(is.na(data$restricted_areas), 0, 1)
data$restricted_areas <- NULL
data$has_pre_ico_price_usd <- ifelse(is.na(data$pre_ico_price_usd), 0, 1)
data$pre_ico_price_usd <- NULL
### mean and mode imputation
most_common <- function(x) {
  ux <- unique(na.omit(x))
  ux[which.max(tabulate(match(x, ux)))]
}
data$accepting[is.na(data$accepting)] <- most_common(data$accepting)
### mice imputation
missing_data <- colSums(is.na(data))
missing_data
impute_variables <- data[, c("price_usd", "rating", "teamsize", "distributed_in_ico", "sold_tokens", 
                             "token_for_sale", "ERC20")]
data$price_usd <- as.numeric(data$price_usd)
data$rating <- as.numeric(data$rating)
data$teamsize <- as.numeric(data$teamsize)
data$distributed_in_ico <- as.numeric(data$distributed_in_ico)
data$sold_tokens <- as.numeric(data$sold_tokens)
data$token_for_sale <- as.numeric(data$token_for_sale)
data$ERC20 <- as.factor(data$ERC20)
impute_variables$distributed_in_ico <- log1p(impute_variables$distributed_in_ico)
impute_variables$token_for_sale <- log1p(impute_variables$token_for_sale)
impute_variables$sold_tokens <- as.numeric(impute_variables$sold_tokens)
impute_variables$sold_tokens <- log1p(impute_variables$sold_tokens)
ini <- mice(impute_variables, maxit = 0)
ini$predictorMatrix["rating", ] <- 0
ini$predictorMatrix["rating", "teamsize"] <- 1
impute <- mice(impute_variables, method = "pmm", m = 30, seed = 123)
completed_data <- complete(impute, 1)
data[, c("price_usd", "rating", "teamsize", "distributed_in_ico", "token_for_sale", "sold_tokens",
         "ERC20")] <- completed_data
missing_data <- colSums(is.na(data))
missing_data

## scaling numerical variables
numeric_columns <- sapply(data, is.numeric)
numeric_columns["success"] <- FALSE
data_scaled <- data
data_scaled[, numeric_columns] <- scale(data[, numeric_columns])
missing_data <- colSums(is.na(data_scaled))
missing_data

# MODELLING

## split data
data_scaled$success <- as.factor(data_scaled$success)
data_scaled$country <- factor(data_scaled$country) 
country_counts <- table(data_scaled$country)
rare_countries <- names(country_counts[country_counts < 13])  
data_scaled$country <- as.character(data_scaled$country)
data_scaled$country[data_scaled$country %in% rare_countries] <- "Other"
data_scaled$country <- factor(data_scaled$country)
accepting_grepl <- function(x) {
  x <- tolower(x)  
  if (grepl("fiat", x) & grepl("btc", x) & grepl("eth", x)) {
    return("Fiat_BTC_ETH")
  } else if (grepl("fiat", x)) {
    return("Fiat")
  } else if (grepl("btc", x) & grepl("eth", x)) {
    return("BTC_ETH")
  } else if (grepl("btc", x)) {
    return("BTC")
  } else if (grepl("eth", x)) {
    return("ETH")
  } else {
    return("Other")
  }
}
data_scaled$accepting <- sapply(data_scaled$accepting, accepting_grepl)
data_scaled$accepting <- factor(data_scaled$accepting)
trainIndex <- createDataPartition(data_scaled$success, p = 0.8, list = FALSE)
train_data <- data_scaled[trainIndex, ]
test_data <- data_scaled[-trainIndex, ]
all_countries <- union(levels(factor(train_data$country)), levels(factor(test_data$country)))
train_data$country <- factor(train_data$country, levels = all_countries)
test_data$country <- factor(test_data$country, levels = all_countries)
all_accepting <- union(levels(factor(train_data$accepting)), levels(factor(test_data$accepting)))
train_data$accepting <- factor(train_data$accepting, levels = all_accepting)
test_data$accepting <- factor(test_data$accepting, levels = all_accepting)
train_data$price_usd <- as.numeric(as.character(train_data$price_usd))
test_data$price_usd <- as.numeric(as.character(test_data$price_usd))
train_data$price_usd[is.na(train_data$price_usd)] <- mean(train_data$price_usd, na.rm = TRUE)
test_data$price_usd[is.na(test_data$price_usd)] <- mean(train_data$price_usd, na.rm = TRUE) 
str(train_data)
colSums(is.na(train_data))
colSums(is.na(test_data))

## logistic regression
log_model <- glm(success ~ ., data = train_data, family = binomial)
log_pred_probs <- predict(log_model, newdata = test_data, type = "response")
log_pred <- ifelse(log_pred_probs > 0.5, 1, 0)

## decision tree
tree_model <- rpart(success ~ ., data = train_data, method = "class") 
tree_pred_probs <- predict(tree_model, newdata = test_data, type = "prob")[,2]
tree_pred <- ifelse(tree_pred_probs > 0.5, 1, 0)

## random forest 
rf_model <- randomForest(success ~ ., data = train_data, ntree = 500, importance = TRUE)
rf_pred_probs <- predict(rf_model, newdata = test_data, type = "prob")[,2]
rf_pred <- ifelse(rf_pred_probs > 0.5, 1, 0)

## xgboost
train_matrix <- model.matrix(success ~ . -1, data = train_data)
test_matrix <- model.matrix(success ~ . -1, data = test_data)
train_label <- as.numeric(train_data$success) - 1  
test_label <- as.numeric(test_data$success) - 1    
data_train <- xgb.DMatrix(data = train_matrix, label = train_label)
data_test <- xgb.DMatrix(data = test_matrix, label = test_label)
xgb_model <- xgboost(data = data_train, 
                     objective = "binary:logistic",
                     nrounds = 100)
xgb_pred_probs <- predict(xgb_model, data_test)
xgb_pred <- ifelse(xgb_pred_probs > 0.5, 1, 0)


# EVALUATION
# logistic regression
conf_log <- confusionMatrix(as.factor(log_pred), as.factor(test_label), positive = "1")
TP_log <- conf_log$table[2, 2]
TN_log <- conf_log$table[1, 1]
FP_log <- conf_log$table[1, 2]
FN_log <- conf_log$table[2, 1]
accuracy_log <- (TP_log + TN_log) / (TP_log + TN_log + FP_log + FN_log)
error_rate_log <- 1 - accuracy_log
sensitivity_log <- TP_log / (TP_log + FN_log)
specificity_log <- TN_log / (TN_log + FP_log)
precision_log <- TP_log / (TP_log + FP_log)
recall_log <- sensitivity_log
f1_log <- 2 * precision_log * recall_log / (precision_log + recall_log)
log_roc <- roc(test_label, log_pred_probs)
log_auc <- auc(log_roc)

# decision tree
conf_tree <- confusionMatrix(as.factor(tree_pred), as.factor(test_label), positive = "1")
TP_tree <- conf_tree$table[2, 2]
TN_tree <- conf_tree$table[1, 1]
FP_tree <- conf_tree$table[1, 2]
FN_tree <- conf_tree$table[2, 1]
accuracy_tree <- (TP_tree + TN_tree) / (TP_tree + TN_tree + FP_tree + FN_tree)
error_rate_tree <- 1 - accuracy_tree
sensitivity_tree <- TP_tree / (TP_tree + FN_tree)
specificity_tree <- TN_tree / (TN_tree + FP_tree)
precision_tree <- TP_tree / (TP_tree + FP_tree)
recall_tree <- sensitivity_tree
f1_tree <- 2 * precision_tree * recall_tree / (precision_tree + recall_tree)
tree_roc <- roc(test_label, tree_pred_probs)
tree_auc <- auc(tree_roc)

# random forest
conf_rf <- confusionMatrix(as.factor(rf_pred), as.factor(test_label), positive = "1")
TP_rf <- conf_rf$table[2, 2]
TN_rf <- conf_rf$table[1, 1]
FP_rf <- conf_rf$table[1, 2]
FN_rf <- conf_rf$table[2, 1]
accuracy_rf <- (TP_rf + TN_rf) / (TP_rf + TN_rf + FP_rf + FN_rf)
error_rate_rf <- 1 - accuracy_rf
sensitivity_rf <- TP_rf / (TP_rf + FN_rf)
specificity_rf <- TN_rf / (TN_rf + FP_rf)
precision_rf <- TP_rf / (TP_rf + FP_rf)
recall_rf <- sensitivity_rf
f1_rf <- 2 * precision_rf * recall_rf / (precision_rf + recall_rf)
rf_roc <- roc(test_label, rf_pred_probs)
rf_auc <- auc(rf_roc)

# xgboost
conf_xgb <- confusionMatrix(as.factor(xgb_pred), as.factor(test_label), positive = "1")
TP_xgb <- conf_xgb$table[2, 2]
TN_xgb <- conf_xgb$table[1, 1]
FP_xgb <- conf_xgb$table[1, 2]
FN_xgb <- conf_xgb$table[2, 1]
accuracy_xgb <- (TP_xgb + TN_xgb) / (TP_xgb + TN_xgb + FP_xgb + FN_xgb)
error_rate_xgb <- 1 - accuracy_xgb
sensitivity_xgb <- TP_xgb / (TP_xgb + FN_xgb)
specificity_xgb <- TN_xgb / (TN_xgb + FP_xgb)
precision_xgb <- TP_xgb / (TP_xgb + FP_xgb)
recall_xgb <- sensitivity_xgb
f1_xgb <- 2 * precision_xgb * recall_xgb / (precision_xgb + recall_xgb)
xgb_roc <- roc(test_label, xgb_pred_probs)
xgb_auc <- auc(xgb_roc)

# results
results_df <- data.frame(
  Model = c("Logistic", "Decision Tree", "Random Forest", "XGBoost"),
  Accuracy = round(c(accuracy_log, accuracy_tree, accuracy_rf, accuracy_xgb), 4),
  Sensitivity = round(c(sensitivity_log, sensitivity_tree, sensitivity_rf, sensitivity_xgb), 4),
  Specificity = round(c(specificity_log, specificity_tree, specificity_rf, specificity_xgb), 4),
  Precision = round(c(precision_log, precision_tree, precision_rf, precision_xgb), 4),
  F1_Score = round(c(f1_log, f1_tree, f1_rf, f1_xgb), 4),
  AUC = round(c(log_auc, tree_auc, rf_auc, xgb_auc), 4)
)

print(results_df)