# Model Artifacts

This folder contains serialized machine learning artifacts used for deployment.

## Artifacts

### final_ai_detector.pkl

Serialized machine learning pipeline containing:

- Text preprocessing
- Word TF-IDF vectorizer
- Character TF-IDF vectorizer
- Logistic Regression classifier

### decision_threshold.pkl

Stores the optimized classification threshold used during inference.

### model_metadata.pkl

Stores model configuration and validation metadata.

## Deployment

Load artifacts using Joblib:

```python
import joblib

model = joblib.load("final_ai_detector.pkl")
threshold = joblib.load("decision_threshold.pkl")
metadata = joblib.load("model_metadata.pkl")
