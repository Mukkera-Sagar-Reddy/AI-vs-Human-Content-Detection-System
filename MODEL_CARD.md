# Model Card

## Model Overview

### Model Name

AI vs Human Content Detection System

### Version

v1.0

### Status

Production Ready

### Model Type

Supervised Binary Text Classification

### Primary Objective

Classify textual content as either:

* AI Generated
* Human Written

### Business Use Case

The model was developed to support content moderation workflows by automatically identifying AI-generated content before publication.

Potential applications include:

* Content moderation
* Editorial review workflows
* Academic integrity screening
* Digital publishing platforms
* Content authenticity verification

---

# Model Architecture

## Feature Engineering

### Word-Level Features

Technique:

* TF-IDF Vectorization

Configuration:

* N-Gram Range: (1,2)
* Maximum Features: 5,000
* Stop Words Removal: Enabled
* Sublinear TF Scaling: Enabled

### Character-Level Features

Technique:

* Character TF-IDF Vectorization

Configuration:

* Analyzer: char_wb
* N-Gram Range: (3,5)
* Maximum Features: 10,000
* Sublinear TF Scaling: Enabled

### Feature Combination

Word and Character features are combined using FeatureUnion.

---

# Classification Model

Algorithm:

Logistic Regression

Configuration:

* Class Weight: Balanced
* Maximum Iterations: 3000
* Regularization Parameter (C): 1.0

---

# Decision Threshold

Default Probability Threshold:

0.44

Classification Logic:

* Probability ≥ 0.44 → AI Generated
* Probability < 0.44 → Human Written

---

# Training Dataset

Total Records:

2,000

Classes:

* Human Written
* AI Generated

Domains:

* Academic
* News
* Social

Training Split:

1,592 records

Testing Split:

408 records

Split Method:

GroupShuffleSplit

Purpose:

Prevent duplicate text leakage between training and testing datasets.

---

# Validation Strategy

## Internal Validation

Testing performed using unseen holdout data.

### Results

| Metric    | Score   |
| --------- | ------- |
| Accuracy  | 100.00% |
| Precision | 100.00% |
| Recall    | 100.00% |
| F1 Score  | 100.00% |

---

## External Validation

Validation performed using independently collected content generated from:

### Human Sources

* Academic Writing
* News Content
* Social Content

### AI Sources

* GPT
* Gemini

Total External Samples:

39

### Results

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 82.05% |
| Precision | 81.82% |
| Recall    | 91.67% |
| F1 Score  | 86.47% |

---

# Deployment Artifacts

Serialized Files:

* final_ai_detector.pkl
* decision_threshold.pkl
* model_metadata.pkl

Serialization Framework:

Joblib

---

# Known Limitations

### Language Support

Currently validated only for English language content.

### Short Texts

Performance may degrade on extremely short content samples.

### Emerging LLMs

Future language models may produce writing styles not represented in the current training data.

### Domain Drift

Changes in writing patterns over time may impact performance.

---

# Ethical Considerations

The model should not be used as the sole decision-making mechanism for:

* Academic misconduct investigations
* Employment decisions
* Legal proceedings

Human review is recommended for high-impact decisions.

---

# Monitoring Recommendations

### Monthly

Review prediction quality.

### Quarterly

Validate using newly collected samples.

### Semi-Annual

Assess retraining requirements.

### Annual

Conduct full model performance review.

---

# Repository

This model is provided for educational, research, and portfolio purposes.

Contributions, enhancements, and validation improvements are welcome through Pull Requests.

---

# Author

M. Sagar Reddy

Data Science and AI | Machine Learning | NLP

Version: 1.0
