#data input
library(readr)
library(dplyr)
library(MASS)        # lda, qda
library(glmnet)
library(randomForest)
library(pls)         # pcr/plsr
library(splines)
library(caret)
library(e1071)       # svm
set.seed(1)


#get data
data <- read_csv("credit+approval/crx.data")


vars <- c("Gender","Age","Debt","MaritalStatus","BankCustomer","EducationLevel",
          "Ethnicity","YearsEmployed","PriorDefault","Employed","CreditScore",
          "DriversLicense","Citizen","ZipCode","Income","Approved")
names(data) <- vars


# Clean + encode like your first script (but with clean names)
data[data == "?"] <- NA
data <- na.omit(data)

# Binary encodings
data <- data %>%
  mutate(
    Gender         = ifelse(Gender=="a",1,0),
    PriorDefault   = ifelse(PriorDefault=="t",1,0),
    Employed       = ifelse(Employed=="t",1,0),
    DriversLicense = ifelse(DriversLicense=="t",1,0),
    Approved       = ifelse(Approved=="+",1,0)
  )

# Make sure numeric columns are numeric
data <- data %>%
  mutate(
    Age           = as.numeric(Age),
    Debt          = as.numeric(Debt),
    YearsEmployed = as.numeric(YearsEmployed),
    CreditScore   = as.numeric(CreditScore),
    Income        = as.numeric(Income),
    ZipCode       = as.numeric(ZipCode)
  )

# Zip buckets then drop original
data <- data %>%
  mutate(
    ZipCode1 = as.integer(ZipCode <= 73),
    ZipCode2 = as.integer(ZipCode > 73 & ZipCode <= 160),
    ZipCode3 = as.integer(ZipCode > 160 & ZipCode <= 272),
    ZipCode4 = as.integer(ZipCode > 272)
  ) %>%
  dplyr::select(-ZipCode)

# One-hot categoricals (same levels as your first script), then drop originals
data <- data %>%
  mutate(Citizen_g = as.integer(Citizen=="g"),
         Citizen_p = as.integer(Citizen=="p")) %>%
  dplyr::select(-Citizen)

data <- data %>%
  mutate(MaritalStatus_u = as.integer(MaritalStatus=="u"),
         MaritalStatus_y = as.integer(MaritalStatus=="y"),
         MaritalStatus_l = as.integer(MaritalStatus=="l")) %>%
  dplyr::select(-MaritalStatus)

data <- data %>%
  mutate(BankCustomer_g = as.integer(BankCustomer=="g"),
         BankCustomer_p = as.integer(BankCustomer=="p")) %>%
  dplyr::select(-BankCustomer)

for (lv in c("c","d","cc","i","j","k","m","r","q","w","x","e","aa")) {
  data[[paste0("EducationLevel_", lv)]] <- as.integer(data$EducationLevel == lv)
}
data <- dplyr::select(data, -EducationLevel)

for (lv in c("v","h","bb","j","n","z","dd","ff")) {
  data[[paste0("Ethnicity_", lv)]] <- as.integer(data$Ethnicity == lv)
}
data <- dplyr::select(data, -Ethnicity)


# split first
train_idx <- createDataPartition(factor(data$Approved, levels = c(0,1)), p = 0.8, list = FALSE)
train_data <- data[train_idx, ]
test_data  <- data[-train_idx, ]

# scale only numeric continuous columns, using TRAIN stats
num_cols <- c("Age","Debt","YearsEmployed","CreditScore","Income")

scaler <- scale(train_data[, num_cols])                  # fit on train
train_data[, num_cols] <- scaler
test_data[,  num_cols] <- scale(test_data[, num_cols],
                                center = attr(scaler, "scaled:center"),
                                scale  = attr(scaler, "scaled:scale"))

# targets
train_y <- factor(train_data$Approved, levels = c(0,1))
test_y  <- factor(test_data$Approved,  levels = c(0,1))


confMat <- function(fit, newdata = test_data, truth = test_y, thr = 0.5) {
  probs <- predict(fit, newdata, type = "response")
  pred  <- factor(as.integer(probs > thr), levels = c(0,1))
  as.numeric(caret::confusionMatrix(pred, truth)$overall["Accuracy"]) * 100
}

confMatLabel <- function(fit, newdata = test_data, truth = test_y) {
  p <- predict(fit, newdata)
  pred <- factor(as.character(p), levels = c(0,1))
  as.numeric(caret::confusionMatrix(pred, truth)$overall["Accuracy"]) * 100
}

# RANDOM FOREST
rf.full <- randomForest(factor(Approved) ~ ., data = train_data,
                        ntree = 500, importance = TRUE)

## 2. View importance
importance_vals <- importance(rf.full)
importance_df <- data.frame(Feature = rownames(importance_vals),
                            MeanDecreaseGini = importance_vals[,"MeanDecreaseGini"])
importance_df <- importance_df[order(-importance_df$MeanDecreaseGini), ]

print(importance_df)
# PriorDefault, Employed and DriversLicense have zero variance

# Chi-square test for PriorDefault, Employed, DriversLicense
chisq.test(table(train_data$PriorDefault, train_data$Approved))
chisq.test(table(train_data$Employed, train_data$Approved))
chisq.test(table(train_data$DriversLicense, train_data$Approved))
# P-value < 0.05, statistically significant, keep the features

# Create the formula
significant <- Approved ~ CreditScore + YearsEmployed + Income + Debt + Age + PriorDefault + Employed + DriversLicense

# Significant features
sig_vars <- c("CreditScore", "YearsEmployed", "Income", "Debt", "Age", "PriorDefault", "Employed", "DriversLicense")

train_sig <- as.matrix(train_data[, sig_vars])
test_sig <- as.matrix(test_data[, sig_vars])

train_all <- as.matrix(train_data[, setdiff(names(train_data), "Approved")])
test_all  <- as.matrix(test_data[,  setdiff(names(test_data),  "Approved")])

train_data_rf <- train_data
test_data_rf <- test_data

# Convert Approved to factor for classification
train_data_rf$Approved <- factor(train_data_rf$Approved, levels = c(0,1))
test_data_rf$Approved <- factor(test_data_rf$Approved, levels = c(0,1))

# Now train Random Forest - it will do CLASSIFICATION
fit.rf <- randomForest(
  significant, 
  data = train_data_rf, 
  ntree = 500, mtry = 3)
confMatLabel(fit.rf)

fit.rf.all <- randomForest(
  Approved ~ ., 
  data = train_data_rf, 
  ntree = 500, mtry = 3)
confMatLabel(fit.rf.all)


# LINEAR REGRESSION
fit.lm.sig <- lm(significant, data = train_data)
confMat(fit.lm.sig)  # uses test_data & test_y by default

fit.lm.all <- lm(Approved ~ ., data = train_data)
confMat(fit.lm.all)

# LOGISTIC REGRESSION (classification)
fit.glm.sig <- glm(significant, data = train_data, family = binomial())
confMat(fit.glm.sig)

fit.glm.all <- glm(Approved ~ ., data = train_data, family = binomial())
confMat(fit.glm.all)

# Linear SVM — significant features
fit.svm.lin.sig <- svm(
  significant, 
  data = train_data_rf,
  kernel = "linear",
  scale = FALSE
)
confMatLabel(fit.svm.lin.sig)

# Linear SVM — all features
fit.svm.lin.all <- svm(
  Approved ~ .,
  data = train_data_rf,
  kernel = "linear",
  scale = FALSE
)
confMatLabel(fit.svm.lin.all)

# Radial SVM — significant features
fit.svm.rbf.sig <- svm(
  significant, 
  data = train_data_rf,
  kernel = "radial",   # gamma defaults to 1/num_features
  scale = FALSE
)
confMatLabel(fit.svm.rbf.sig)

# Radial SVM — all features
fit.svm.rbf.all <- svm(
  Approved ~ .,
  data = train_data_rf,
  kernel = "radial",
  scale = FALSE
)
confMatLabel(fit.svm.rbf.all)

# KNN
# Significant features
fit.knn.sig <- train(
  significant,
  data = train_data_rf,
  method = "knn"
)
confMatLabel(fit.knn.sig, newdata = test_data_rf, truth = test_data_rf$Approved)

# All features
fit.knn.all <- train(
  Approved ~ .,
  data = train_data_rf,
  method = "knn"
)
confMatLabel(fit.knn.all, newdata = test_data_rf, truth = test_data_rf$Approved)


# PLS
# Significant features
fit.pls.sig <- train(significant, data = train_data_rf, method = "pls")
confMatLabel(fit.pls.sig, newdata = test_data_rf, truth = test_data_rf$Approved)


# All features
fit.pls.all <- train(Approved ~ ., data = train_data_rf, method = "pls")
confMatLabel(fit.pls.all, newdata = test_data_rf, truth = test_data_rf$Approved)


# LDA
# Non zero significant features 
significant_nonzero <- Approved ~ CreditScore + YearsEmployed + Income + Debt + Age

fit.lda.sig <- lda(significant_nonzero, data = train_data_rf)
lda_pred <- predict(fit.lda.sig, newdata = test_data_rf)$class
mean(lda_pred == test_data_rf$Approved) * 100

# QDA
# Non zero significant features 
fit.qda.sig <- qda(significant_nonzero, data = train_data_rf)
qda_pred <- predict(fit.qda.sig, newdata = test_data_rf)$class
mean(qda_pred == test_data_rf$Approved) * 100


# Splines
# Significant features
fit.spline.sig <- glm(Approved ~ ns(CreditScore, df=4) + ns(YearsEmployed, df=4) + 
                        ns(Income, df=4) + ns(Debt, df=4) + ns(Age, df=4) + 
                        PriorDefault + Employed + DriversLicense, 
                      data = train_data, family = binomial())
confMat(fit.spline.sig)

# All features - reusing existing variables
fit.spline.all <- glm(
  Approved ~ ns(Age,4) + ns(Debt,4) + ns(YearsEmployed,4) + ns(CreditScore,4) + ns(Income,4) +
    . - Age - Debt - YearsEmployed - CreditScore - Income,
  data = train_data, family = binomial()
)
confMat(fit.spline.all)



# RIDGE
# Convert factor to numeric for comparison
test_y_num <- as.numeric(test_y) - 1  # Convert factor levels (1,2) to (0,1)
train_y_num <- as.numeric(train_y) - 1


#Significant features
fit.ridge.sig  <- cv.glmnet(train_sig, train_y_num, alpha = 0, family = "binomial")
pred.ridge.sig <- as.numeric(predict(fit.ridge.sig, test_sig, s = "lambda.min", type = "response"))
mean((pred.ridge.sig > 0.5) == test_y_num) * 100

# All features
fit.ridge.all  <- cv.glmnet(train_all, train_y_num, alpha = 0, family = "binomial")
pred.ridge.all <- as.numeric(predict(fit.ridge.all, test_all, s = "lambda.min", type = "response"))
mean((pred.ridge.all > 0.5) == test_y_num) * 100

# LASSO
# Significant features
fit.lasso.sig  <- cv.glmnet(train_sig, train_y_num, alpha = 1, family = "binomial")
pred.lasso.sig <- as.numeric(predict(fit.lasso.sig, test_sig, s = "lambda.min", type = "response"))
mean((pred.lasso.sig > 0.5) == test_y_num) * 100

# All features
fit.lasso.all  <- cv.glmnet(train_all, train_y_num, alpha = 1, family = "binomial")a
pred.lasso.all <- as.numeric(predict(fit.lasso.all, test_all, s = "lambda.min", type = "response"))
mean((pred.lasso.all > 0.5) == test_y_num) * 100


# SUMMARY
# ===== Simple collected printout of accuracies =====
pr <- function(label, acc) cat(sprintf("%-32s : %6.2f%%\n", label, acc))

cat("\n=== MODEL ACCURACIES ===\n")

# Linear / Logistic / Splines
pr("Linear Reg (significant)",  confMat(fit.lm.sig))
pr("Linear Reg (all)",          confMat(fit.lm.all))
pr("Logistic Reg (significant)",confMat(fit.glm.sig))
pr("Logistic Reg (all)",        confMat(fit.glm.all))
pr("Splines (significant)",     confMat(fit.spline.sig))
pr("Splines (all)",             confMat(fit.spline.all))

# Random Forest
pr("Random Forest (significant)",
   confMatLabel(fit.rf,     newdata=test_data_rf, truth=test_data_rf$Approved))
pr("Random Forest (all)",
   confMatLabel(fit.rf.all, newdata=test_data_rf, truth=test_data_rf$Approved))

# SVM
pr("SVM Linear (significant)",
   confMatLabel(fit.svm.lin.sig, newdata=test_data_rf, truth=test_data_rf$Approved))
pr("SVM Linear (all)",
   confMatLabel(fit.svm.lin.all, newdata=test_data_rf, truth=test_data_rf$Approved))
pr("SVM RBF (significant)",
   confMatLabel(fit.svm.rbf.sig, newdata=test_data_rf, truth=test_data_rf$Approved))
pr("SVM RBF (all)",
   confMatLabel(fit.svm.rbf.all, newdata=test_data_rf, truth=test_data_rf$Approved))

# KNN
pr("KNN (significant)",
   confMatLabel(fit.knn.sig, newdata=test_data_rf, truth=test_data_rf$Approved))
pr("KNN (all)",
   confMatLabel(fit.knn.all, newdata=test_data_rf, truth=test_data_rf$Approved))

# PLS
pr("PLS (significant)",
   confMatLabel(fit.pls.sig, newdata=test_data_rf, truth=test_data_rf$Approved))
pr("PLS (all)",
   confMatLabel(fit.pls.all, newdata=test_data_rf, truth=test_data_rf$Approved))

# LDA / QDA (nonzero significant set)
pr("LDA (significant-nonzero)",
   mean(predict(fit.lda.sig, newdata=test_data_rf)$class == test_data_rf$Approved) * 100)
pr("QDA (significant-nonzero)",
   mean(predict(fit.qda.sig, newdata=test_data_rf)$class == test_data_rf$Approved) * 100)

# Ridge / Lasso (glmnet)
pr("Ridge (significant)",
   mean((as.numeric(predict(fit.ridge.sig, test_sig,  s="lambda.min", type="response")) > 0.5) == (as.numeric(test_y)-1)) * 100)
pr("Ridge (all)",
   mean((as.numeric(predict(fit.ridge.all, test_all, s="lambda.min", type="response")) > 0.5) == (as.numeric(test_y)-1)) * 100)
pr("Lasso (significant)",
   mean((as.numeric(predict(fit.lasso.sig, test_sig,  s="lambda.min", type="response")) > 0.5) == (as.numeric(test_y)-1)) * 100)
pr("Lasso (all)",
   mean((as.numeric(predict(fit.lasso.all, test_all, s="lambda.min", type="response")) > 0.5) == (as.numeric(test_y)-1)) * 100)