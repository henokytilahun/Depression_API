import numpy as np
import pandas as pd
import shap
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import tensorflow as tf
from tensorflow import keras

# ───────────────────────────────────────────────────────────────────────────────
# 1) Load & preprocess
# ───────────────────────────────────────────────────────────────────────────────
df = pd.read_csv('Student_Depression_Dataset.csv')

# drop unused, reset id
df['id'] = np.arange(len(df))
df['CGPA'] = df['CGPA'].apply(lambda x: (x/10)*4)
df.rename(columns={
    'Have you ever had suicidal thoughts ?': 'Suicidal Thoughts',
    'Family History of Mental Illness': 'Family History'
}, inplace=True)

# encode categoricals
cat_cols = [
    'Degree','City','Gender',
    'Dietary Habits','Sleep Duration',
    'Suicidal Thoughts','Family History'
]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# build feature matrix + target
X = df.drop(['id','Profession','Depression'], axis=1)
feature_names = X.columns.tolist()
y = df['Depression'].astype(np.float32)

# fill + scale
X = X.fillna(X.mean())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

# ───────────────────────────────────────────────────────────────────────────────
# 2) Initial split + model for SHAP
# ───────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

def build_model(input_dim):
    return keras.Sequential([
        keras.layers.BatchNormalization(input_shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.BatchNormalization(), keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.BatchNormalization(), keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.BatchNormalization(), keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])

initial_model = build_model(X_train.shape[1])
initial_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
initial_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# ───────────────────────────────────────────────────────────────────────────────
# 3) SHAP feature importance
# ───────────────────────────────────────────────────────────────────────────────
def pred_fn(x): 
    return initial_model.predict(x).flatten()

background = X_train[np.random.choice(len(X_train), 100, replace=False)]
explainer = shap.KernelExplainer(pred_fn, background)
shap_vals = explainer.shap_values(X_test[:100], nsamples=200)

mean_abs_shap = np.abs(shap_vals).mean(axis=0)
importance_df = (
    pd.DataFrame({
        'Feature': feature_names,
        'Mean|SHAP|': mean_abs_shap
    })
    .sort_values('Mean|SHAP|', ascending=False)
    .reset_index(drop=True)
)

# ───────────────────────────────────────────────────────────────────────────────
# 4) Select top K features
# ───────────────────────────────────────────────────────────────────────────────
TOP_K = 5
top_features = importance_df['Feature'].head(TOP_K).tolist()

# persist for your FastAPI app
with open('selected_features.json','w') as f:
    json.dump([f.lower().replace(' ', '_') for f in top_features], f, indent=2)


# ───────────────────────────────────────────────────────────────────────────────
# 5) Retrain on reduced features
# ───────────────────────────────────────────────────────────────────────────────
# rebuild X_scaled as DataFrame for easy column selection
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
X_red = X_scaled_df[top_features].values

scaler = StandardScaler()
Xr_scaled = scaler.fit_transform(X_red)
pickle.dump(scaler, open('scaler.pkl','wb'))


Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_red, y, test_size=0.2, random_state=42
)

model = build_model(len(top_features))
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy','AUC','Precision','Recall']
)
history = model.fit(
    Xr_train, yr_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    verbose=1
)

# final evaluation
results = model.evaluate(Xr_test, yr_test, verbose=0)
print("Reduced-model test results:", dict(zip(model.metrics_names, results)))

# ───────────────────────────────────────────────────────────────────────────────
# 6) Save artifacts
# ───────────────────────────────────────────────────────────────────────────────
model.save('model.h5')
with open('scaler.pkl','wb') as f:     pickle.dump(scaler,f)
for col,le in encoders.items():
    fn = f"le_{col.replace(' ','_').lower()}.pkl"
    with open(fn, 'wb') as encoder_file:
        pickle.dump(le, encoder_file)

print("Top features:", top_features)