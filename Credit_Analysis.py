import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

# Load & name the columns
cols = ["Gender","Age","Debt","MaritalStatus","BankCustomer","EducationLevel",
        "Ethnicity","YearsEmployed","PriorDefault","Employed","CreditScore",
        "DriversLicense","Citizen","ZipCode","Income","Approved"]

df = pd.read_csv("credit+approval/crx.data", header=None, names=cols, na_values=["?"])
df = df.dropna().copy()

# Binary encodings
df["Gender"]         = (df["Gender"] == "a").astype(int)
df["PriorDefault"]   = (df["PriorDefault"] == "t").astype(int)
df["Employed"]       = (df["Employed"] == "t").astype(int)
df["DriversLicense"] = (df["DriversLicense"] == "t").astype(int)
df["Approved"]       = (df["Approved"] == "+").astype(int)

# Ensure numeric columns
num_cols = ["Age","Debt","YearsEmployed","CreditScore","Income","ZipCode"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna().copy()

# Zip buckets then drop ZipCode
df["ZipCode1"] = (df["ZipCode"] <= 73).astype(int)
df["ZipCode2"] = ((df["ZipCode"] > 73) & (df["ZipCode"] <= 160)).astype(int)
df["ZipCode3"] = ((df["ZipCode"] > 160) & (df["ZipCode"] <= 272)).astype(int)
df["ZipCode4"] = (df["ZipCode"] > 272).astype(int)
df = df.drop(columns=["ZipCode"])

# Citizen g/p
df["Citizen_g"] = (df["Citizen"] == "g").astype(int)
df["Citizen_p"] = (df["Citizen"] == "p").astype(int)
df = df.drop(columns=["Citizen"])

# MaritalStatus u/y/l
df["MaritalStatus_u"] = (df["MaritalStatus"] == "u").astype(int)
df["MaritalStatus_y"] = (df["MaritalStatus"] == "y").astype(int)
df["MaritalStatus_l"] = (df["MaritalStatus"] == "l").astype(int)
df = df.drop(columns=["MaritalStatus"])

# BankCustomer g/p
df["BankCustomer_g"] = (df["BankCustomer"] == "g").astype(int)
df["BankCustomer_p"] = (df["BankCustomer"] == "p").astype(int)
df = df.drop(columns=["BankCustomer"])

# EducationLevel
for lv in ["c","d","cc","i","j","k","m","r","q","w","x","e","aa"]:
    df[f"EducationLevel_{lv}"] = (df["EducationLevel"] == lv).astype(int)
df = df.drop(columns=["EducationLevel"])

# Ethnicity
for lv in ["v","h","bb","j","n","z","dd","ff"]:
    df[f"Ethnicity_{lv}"] = (df["Ethnicity"] == lv).astype(int)
df = df.drop(columns=["Ethnicity"])

# Train/test split (80/20)
data = df.drop(columns=["Approved"])
target = df["Approved"].astype(int)

train_features, test_features, train_labels, test_labels = train_test_split(
    data, target, test_size=0.2, stratify=target, random_state=1
)

# Scale ONLY the 5 continuous numeric columns (train stats)
numCols = ["Age","Debt","YearsEmployed","CreditScore","Income"]

scaler = StandardScaler().fit(train_features[numCols])
train_features_scaled = train_features.copy()
test_features_scaled  = test_features.copy()

train_features_scaled[numCols] = scaler.transform(train_features[numCols])
test_features_scaled[numCols]  = scaler.transform(test_features[numCols])


# Significant feature subset (same as R)
sig_vars = ["CreditScore","YearsEmployed","Income","Debt","Age","PriorDefault","Employed","DriversLicense"]

train_sig = train_features_scaled[sig_vars].to_numpy()
test_sig = test_features_scaled[sig_vars].to_numpy()

train_all = train_features_scaled.to_numpy()
test_all = test_features_scaled.to_numpy()

corr_matrix = df.corr()

# Get correlations with 'Approved' column
approved_corr = corr_matrix['Approved'].abs().sort_values(ascending=False)

# Filter for correlations > 0.2 (excluding 'Approved' itself)
high_corr = approved_corr[approved_corr > 0.2][1:]


plt.figure(figsize=(10, 6))
plt.bar(high_corr.index, high_corr.values)
plt.title('Variables with Correlation > 0.2 to Approved')
plt.xlabel('Variables')
plt.ylabel('Absolute Correlation')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Train Random Forest on all features to get feature importance
rf_importance = RandomForestClassifier(n_estimators=500, random_state=1)
rf_importance.fit(train_all, train_labels)

# Get feature names and importance scores
feature_names = train_features_scaled.columns
importance_scores = rf_importance.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance_scores
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df['Feature'][:15], feature_importance_df['Importance'][:15])  # Show top 15
plt.title('Random Forest Feature Importance (Increasing Node Purity)')
plt.xlabel('Increasing Node Purity')
plt.ylabel('Variables')
plt.gca().invert_yaxis()  # Highest importance at top
plt.tight_layout()
plt.show()


# Helper functions
def Accuracy(label, acc):
    print(f"{label:<32s} : {acc:6.2f}%")

def AccLabels(model, X, y):
    return accuracy_score(y, model.predict(X)) * 100

def AccScores(scores, y):  # for regression-style outputs
    return accuracy_score(y, (scores > 0.5).astype(int)) * 100

def plot_roc_svm(models, names, X, y, title):
    plt.figure(figsize=(7, 6))
    for model, name in zip(models, names):
        # SVC exposes decision_function (signed distance to hyperplane)
        scores = model.decision_function(X)
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0,1], [0,1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_roc_logreg(models, names, Xs, y, title):
    plt.figure(figsize=(7,6))
    for model, name, X in zip(models, names, Xs):
        # Logistic regression: use predicted probabilities for class 1
        scores = model.predict_proba(X)[:,1]
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_calibration(models, names, Xs, y, title):
    plt.figure(figsize=(7,6))
    for model, name, X in zip(models, names, Xs):
        probs = model.predict_proba(X)[:,1]
        fraction_of_positives, mean_predicted_value = calibration_curve(y, probs, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "o-", label=name)

    # Ideal diagonal
    plt.plot([0,1],[0,1],"k--", lw=1)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Proportion Approved")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Baseline (majority class)
base = test_labels.value_counts(normalize=True).max() * 100
print(f"Baseline (majority-class)       : {base:6.2f}%\n")


# MODELS

# Random Forest (mtry ~ 3 in R -> max_features=3)
# Forest will have 500 trees and each split in a tree, only 3 features are considered
rf_sig = RandomForestClassifier(n_estimators=500, max_features=3, random_state=1).fit(train_sig, train_labels)
rf_all = RandomForestClassifier(n_estimators=500, max_features=3, random_state=1).fit(train_all, train_labels)

# Linear Regression (then 0.5 threshold)
# straight line or hyperplane in higher dimensions to minimize squared error between predicition and actual values
lm_sig = LinearRegression().fit(train_sig, train_labels)
lm_all = LinearRegression().fit(train_all, train_labels)

pred_sig = lm_sig.predict(test_sig)

plt.figure(figsize=(8,5))
plt.hist(pred_sig[test_labels==0], bins=20, alpha=0.6, label="Not Approved (0)")
plt.hist(pred_sig[test_labels==1], bins=20, alpha=0.6, label="Approved (1)")
plt.axvline(0.5, color="red", linestyle="--", label="0.5 threshold")
plt.xlabel("Predicted Value")
plt.ylabel("Count")
plt.title("Linear Regression Predictions (Significant Variables)")
plt.legend()
plt.show()

pred_all = lm_all.predict(test_all)

plt.figure(figsize=(8,5))
plt.hist(pred_sig[test_labels==0], bins=20, alpha=0.6, label="Not Approved (0)")
plt.hist(pred_sig[test_labels==1], bins=20, alpha=0.6, label="Approved (1)")
plt.axvline(0.5, color="red", linestyle="--", label="0.5 threshold")
plt.xlabel("Predicted Value")
plt.ylabel("Count")
plt.title("Linear Regression Predictions (All Variables)")
plt.legend()
plt.show()


# Logistic Regression (plain)
# Takes input features and makes a linear combination: z = beta_0 + beta_1 * x_1 + beta_2 * x_2...
# Passes z to a sigmoid function and squashes any number into a probability between 0 and 1
# p >= 0.5 -> class 1 (yes) else class 0 (no)
log_sig = LogisticRegression(solver="liblinear", max_iter=1000).fit(train_sig, train_labels)
log_all = LogisticRegression(solver="liblinear", max_iter=1000).fit(train_all, train_labels)

plot_roc_logreg(
    models=[log_sig, log_all],
    names=["Logistic (significant vars)", "Logistic (all vars)"],
    Xs=[test_sig, test_all],
    y=test_labels,
    title="Logistic Regression ROC Curves"
)

# SVMs

# boundary is a straight line and C controls how strict the margin is
# Large C -> small margin so fewer misclassification
# Small C -> large margin, allows more misclassification (overfitting)
svm_lin_sig = SVC(kernel="linear", C=1).fit(train_sig, train_labels)
svm_lin_all = SVC(kernel="linear", C=1).fit(train_all, train_labels)

# rbf is radial basis function (Gaussian Kernel)
# maps data into higher dimensions so classes that aren't linearly separable become separable
# gamma defines how far the influence of a single training example reaches
# low values means influence extends far and high values meaning the influence is limited to points very close to the training example
svm_rbf_sig = SVC(kernel="rbf", C=1, gamma="scale").fit(train_sig, train_labels)
svm_rbf_all = SVC(kernel="rbf", C=1, gamma="scale").fit(train_all, train_labels)

# Significant features
plot_roc_svm(
    models=[svm_lin_sig, svm_rbf_sig],
    names=["Linear (sig)", "RBF (sig)"],
    X=test_sig, y=test_labels,
    title="SVM ROC (Significant Features)"
)

# All features
plot_roc_svm(
    models=[svm_lin_all, svm_rbf_all],
    names=["Linear (all)", "RBF (all)"],
    X=test_all, y=test_labels,
    title="SVM ROC (All Features)"
)

# KNN
# For each prediction look at k closest neighbors, take majority vote of their labels, assign the new point to that class

k_values = range(1, 41)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_sig, train_labels)

    # Use validation accuracy instead of cross-validation
    accuracy = knn.score(test_sig, test_labels)
    accuracies.append(accuracy)

optimal_k = k_values[np.argmax(accuracies)]
max_accuracy = max(accuracies)

print(f"Best accuracy K = {optimal_k}, which accuracy {accuracy:.3f}")
# Then plot normally
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, 'o-', markersize=6, linewidth=1)
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('K-NN Accuracy vs K Value')
plt.grid(True, alpha=0.3)
plt.show()


knn_sig = KNeighborsClassifier(n_neighbors=optimal_k).fit(train_sig, train_labels)
knn_all = KNeighborsClassifier(n_neighbors=optimal_k).fit(train_all, train_labels)

# Find optimal K
optimal_k = k_values[np.argmax(accuracies)]
max_accuracy = max(accuracies)

print(f"\nOptimal K = {optimal_k} with accuracy = {max_accuracy:.3f}")


# PLS (regression, then 0.5 threshold) combines aspects of PCA
# Sometimes features are many and highly correlated (multicolinearlity)
# PLS creates new components (linear combinations of the original features)
# Instead of regressing on all raw features, regress on a small number of latent components
# It works well with many correlated predictors

# Test 1..7 components on significant features
mse = []
n_components = range(1, 8)

for n in n_components:
    pls = PLSRegression(n_components=n)
    y_pred = cross_val_predict(pls, train_sig, train_labels, cv=5)
    mse.append(mean_squared_error(train_labels, y_pred))

# Plot
plt.figure(figsize=(8,6))
plt.plot(n_components, mse, marker='o', linestyle='-', color="blue")
plt.title("Approved")
plt.xlabel("number of components")
plt.ylabel("MSEp")
plt.grid(alpha=0.3)

# Add horizontal line at minimum MSE
plt.axhline(y=min(mse), color='red', linestyle='--')
plt.show()


# number of latent components to use
ncomp_sig = min(3, train_sig.shape[1])
ncomp_all = min(3, train_all.shape[1])
pls_sig = PLSRegression(n_components=ncomp_sig).fit(train_sig, train_labels)
pls_all = PLSRegression(n_components=ncomp_all).fit(train_all, train_labels)

# LDA / QDA on non-zero significant numerics

nonZero_sig = ["CreditScore","YearsEmployed","Income","Debt","Age"]
train_nz = train_features_scaled[nonZero_sig].to_numpy()
test_nz = test_features_scaled[nonZero_sig].to_numpy()

# decision boundary is linear (line, plane or hyperplane)
# will find the linear combination of features that best separates the classes
lda_sig = LDA().fit(train_nz, train_labels)

# decision boundary is quadratic
# can overfit if data is small or noisy
# reg_param is a small regularization term to stabilize covariance estimates
qda_sig = QDA(reg_param=1e-4).fit(train_nz, train_labels)


lda_preds = lda_sig.predict(test_nz)
qda_preds = qda_sig.predict(test_nz)

# Confusion matrices
cm_lda = confusion_matrix(test_labels, lda_preds)
cm_qda = confusion_matrix(test_labels, qda_preds)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# LDA heatmap
sns.heatmap(cm_lda, annot=True, fmt='d', cmap="Blues", cbar=False,
            xticklabels=["Not Approved", "Approved"],
            yticklabels=["Not Approved", "Approved"], ax=axes[0])
axes[0].set_title("LDA Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# QDA heatmap
sns.heatmap(cm_qda, annot=True, fmt='d', cmap="Greens", cbar=False,
            xticklabels=["Not Approved", "Approved"],
            yticklabels=["Not Approved", "Approved"], ax=axes[1])
axes[1].set_title("QDA Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()


# Ridge / Lasso (logistic);
# Squares of coefficients are penalized

# Coefficients get shrunk toward zero but rarely become exactly zero
# Keeps all features, but reduces their impact if they're not useful
ridge_sig = LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000).fit(train_sig, train_labels)
ridge_all = LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000).fit(train_all, train_labels)

plot_roc_logreg(
    models=[ridge_sig, ridge_all],
    names=["Ridge Logistic (sig vars)", "Ridge Logistic (all vars)"],
    Xs=[test_sig, test_all],
    y=test_labels,
    title="Ridge Logistic Regression ROC Curves"
)


# Absolute values of coefficients are penalized
# Drives some features exactly to 0, so automatic feature selection
lasso_sig = LogisticRegression(penalty="l1", solver="saga", max_iter=5000).fit(train_sig, train_labels)
lasso_all = LogisticRegression(penalty="l1", solver="saga", max_iter=5000).fit(train_all, train_labels)

plot_roc_logreg(
    models=[lasso_sig, lasso_all],
    names=["Lasso Logistic (sig vars)", "Lasso Logistic (all vars)"],
    Xs=[test_sig, test_all],
    y=test_labels,
    title="Lasso Logistic Regression ROC Curves"
)




num_sig = ["CreditScore","YearsEmployed","Income","Debt","Age"]
bin_sig = ["PriorDefault","Employed","DriversLicense"]  # just pass these through


# numeric features: cubic spline basis expansion
# binary features are passed through as is and other features are dropped
# n_knots splits the numeric range into intervals so the curve can bend
# extrapolation outside the knots, so it continues linearly
# include_bias = False, doesnt add an extra intercept column
ct_sig = ColumnTransformer(
    transformers=[
        ("spline", SplineTransformer(degree=3, n_knots=5, extrapolation="linear", include_bias=False), num_sig),
        ("passthrough", "passthrough", bin_sig),
    ],
    remainder="drop"
)
spl_log_sig = make_pipeline(ct_sig, LogisticRegression(solver="liblinear", max_iter=1000)).fit(
    train_features_scaled[sig_vars], train_labels
)



all_other_cols = [c for c in train_features_scaled.columns if c not in numCols]

ct_all = ColumnTransformer(
    transformers=[
        ("spline", SplineTransformer(degree=3, n_knots=5, extrapolation="linear", include_bias=False), numCols),
        ("passthrough", "passthrough", [c for c in all_other_cols if c != "Approved"]), # keep other features (categorical, binary)
    ],
    remainder="drop"
)

spl_log_all = make_pipeline(ct_all, LogisticRegression(solver="liblinear", max_iter=1000)).fit(
    train_features_scaled, train_labels
)

plot_calibration(
    models=[spl_log_sig, spl_log_all],
    names=["Spline (sig vars)", "Spline (all vars)"],
    Xs=[train_features_scaled[sig_vars], train_features_scaled],
    y=train_labels,
    title="Calibration Plot - Splines"
)


# PRINTOUTS
print("\n=== MODEL ACCURACIES ===")
# Linear / Logistic / Splines
Accuracy("Linear Reg (significant)",  AccScores(lm_sig.predict(test_sig), test_labels))
Accuracy("Linear Reg (all)",          AccScores(lm_all.predict(test_all), test_labels))
Accuracy("Logistic Reg (significant)",AccLabels(log_sig, test_sig, test_labels))
Accuracy("Logistic Reg (all)",        AccLabels(log_all, test_all, test_labels))
Accuracy("Splines (significant)",     AccLabels(spl_log_sig, test_features_scaled[sig_vars], test_labels))
Accuracy("Splines (all)",             AccLabels(spl_log_all, test_features_scaled, test_labels))

# Random Forest
Accuracy("Random Forest (significant)",AccLabels(rf_sig, test_sig, test_labels))
Accuracy("Random Forest (all)",        AccLabels(rf_all, test_all, test_labels))

# SVM
Accuracy("SVM Linear (significant)",   AccLabels(svm_lin_sig, test_sig, test_labels))
Accuracy("SVM Linear (all)",           AccLabels(svm_lin_all, test_all, test_labels))
Accuracy("SVM RBF (significant)",      AccLabels(svm_rbf_sig, test_sig, test_labels))
Accuracy("SVM RBF (all)",              AccLabels(svm_rbf_all, test_all, test_labels))

# KNN
Accuracy("KNN (significant)",          AccLabels(knn_sig, test_sig, test_labels))
Accuracy("KNN (all)",                  AccLabels(knn_all, test_all, test_labels))

# PLS (thresholded)
Accuracy("PLS (significant)",          AccScores(pls_sig.predict(test_sig).ravel(), test_labels))
Accuracy("PLS (all)",                  AccScores(pls_all.predict(test_all).ravel(), test_labels))

# LDA / QDA (nonzero significant numerics)
Accuracy("LDA (significant-nonzero)",  AccLabels(lda_sig, test_nz, test_labels))
Accuracy("QDA (significant-nonzero)",  AccLabels(qda_sig, test_nz, test_labels))

# Ridge / Lasso (logistic)
Accuracy("Ridge (significant)",        AccLabels(ridge_sig, test_sig, test_labels))
Accuracy("Ridge (all)",                AccLabels(ridge_all, test_all, test_labels))
Accuracy("Lasso (significant)",        AccLabels(lasso_sig, test_sig, test_labels))
Accuracy("Lasso (all)",                AccLabels(lasso_all, test_all, test_labels))