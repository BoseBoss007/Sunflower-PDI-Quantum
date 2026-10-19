import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create artifacts dir if it doesn't exist
os.makedirs('artifacts/figures', exist_ok=True)

# Load model and data
model = joblib.load('models/classical/random_forest_pdi.joblib')
df = pd.read_csv('artifacts/features/phase1_features.csv')
features = df.drop(columns=['path', 'class', 'pdi_label']).columns

# Get importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Take top 15 features
top_n = 15
top_features = [features[i] for i in indices[:top_n]]
top_importances = [importances[i] for i in indices[:top_n]]

# Create a beautiful matplotlib barplot
plt.figure(figsize=(10, 8))
y_pos = np.arange(len(top_features))

# Create horizontal bars
bars = plt.barh(y_pos, top_importances, align='center', color='teal')
plt.yticks(y_pos, top_features)
plt.gca().invert_yaxis()  # labels read top-to-bottom

plt.title('Top 15 Most Predictive Features for PDI (Random Forest)', fontsize=16, pad=20)
plt.xlabel('Feature Importance Score', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)

# Add value labels to the end of the bars
for i, v in enumerate(top_importances):
    plt.text(v + 0.002, i + 0.1, f'{v:.3f}', color='black', fontsize=10)

plt.tight_layout()
plt.savefig('artifacts/figures/feature_importance.png', dpi=300)
print("Saved feature importance chart to artifacts/figures/feature_importance.png")
