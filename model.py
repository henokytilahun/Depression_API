import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import shap

# 1) Load & preprocess
df = pd.read_csv('Student_Depression_Dataset.csv')

# Replace id with 0..n-1
df['id'] = np.arange(len(df))

# Rescale CGPA from 0–10 to 0–4
df['CGPA'] = df['CGPA'].apply(lambda x: (x / 10) * 4)

# Encode categorical features
le = LabelEncoder()
for col in ['Degree', 'City', 'Gender', 'Dietary Habits', 'Sleep Duration']:
    df[col] = le.fit_transform(df[col])

# Rename and encode the remaining categorical columns
df.rename(columns={
    'Have you ever had suicidal thoughts ?': 'Suicidal Thoughts',
    'Family History of Mental Illness': 'Family History'
}, inplace=True)
df['Suicidal Thoughts'] = le.fit_transform(df['Suicidal Thoughts'])
df['Family History']    = le.fit_transform(df['Family History'])

# 2) Define feature matrix X and target vector y (target is Depression)
X = df.drop(['id', 'Profession', 'Depression'], axis=1)
y = df['Depression']

# Fill missing values with column means and cast to float32
X = X.fillna(X.mean()).astype(np.float32)
y = y.astype(np.float32)

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) Build & train the Keras model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 5) Wrap model for SHAP KernelExplainer
def model_predict(data_matrix: np.ndarray) -> np.ndarray:
    """Return model predicted probability for each sample."""
    return model.predict(data_matrix).flatten()

# 6) Create SHAP explainer with a small background sample
background = X_train.sample(100, random_state=42).values
explainer = shap.KernelExplainer(model_predict, background)

# 7) Compute SHAP values on a subset of the test set
test_subset = X_test.values[:100]
shap_values = explainer.shap_values(test_subset, nsamples=200)

# 8) Beeswarm (summary) plot
shap.summary_plot(
    shap_values,
    features=test_subset,
    feature_names=X.columns,
    show=False
)
plt.tight_layout()
plt.savefig('shap_summary.png')
plt.close()

# 9) Bar chart of mean absolute SHAP values
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Mean(|SHAP|)': mean_abs_shap
}).sort_values('Mean(|SHAP|)', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(importance_df['Feature'], importance_df['Mean(|SHAP|)'])
plt.xticks(rotation=45, ha='right')
plt.title('Mean Absolute SHAP Values')
plt.tight_layout()
plt.savefig('shap_bar.png')
plt.close()

# 10) Print numeric SHAP importances
print("\nMean Absolute SHAP Values:")
print(importance_df)
