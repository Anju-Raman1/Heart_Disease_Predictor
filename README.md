# Heart Disease Predictor (Logistic Regression)

A binary classification project that estimates the probability that a patient has heart disease from 13 medical measurements, plus a small web page that runs the trained model in the browser.

**Live demo:** https://anju-raman1.github.io/Heart_Disease_Predictor/

> **Educational project, not medical advice.** This model must not be used to diagnose anyone or to make health decisions.

## What this project shows

- Classification instead of price prediction: the model outputs a probability, not a number.
- The metrics that matter for classification: accuracy, precision, recall, confusion matrix, ROC-AUC.
- How moving the decision threshold trades missed patients against false alarms.
- A complete pipeline: load, clean, explore, split, compare models with cross-validation, evaluate, interpret, export, deploy.

## Data

UCI Heart Disease dataset, Cleveland subset: 303 patients, 13 features. Six patients with missing values (`ca`, `thal`) were removed, leaving **297 patients** (46% with heart disease). The original target (0 to 4) was turned into a yes/no label: 1 if disease is present, otherwise 0.

## Method

- 80/20 stratified train/test split (237 train, 60 test), `random_state=42`.
- Numeric features scaled with `StandardScaler`; categorical features one-hot encoded. Both steps sit inside a scikit-learn pipeline, so nothing leaks from the validation folds.
- Baseline: always predict the most common class.
- Logistic Regression with `C` in {0.01, 0.1, 1, 10, 100}, chosen by 5-fold cross-validated ROC-AUC on the training set. The test set was kept as a final check.

## Results

Model comparison:

| C | Train accuracy | CV ROC-AUC | Test accuracy | Test recall | Test ROC-AUC |
|---|---|---|---|---|---|
| 0.01 | 0.814 | 0.888 | 0.833 | 0.714 | 0.933 |
| 0.1 | 0.873 | 0.901 | 0.850 | 0.786 | 0.940 |
| **1** | 0.882 | **0.902** | 0.817 | 0.714 | 0.941 |
| 10 | 0.890 | 0.899 | 0.817 | 0.750 | 0.936 |
| 100 | 0.890 | 0.897 | 0.833 | 0.786 | 0.935 |

Selected model (C = 1) on the 60 test patients:

| Metric | Value |
|---|---|
| Accuracy | 0.817 (baseline: 0.533) |
| Precision | 0.870 |
| Recall | 0.714 |
| F1 | 0.784 |
| ROC-AUC | 0.941 |

Confusion matrix: 29 healthy patients correctly cleared, 20 sick patients correctly found, 3 false alarms, **8 missed sick patients** (out of 28).

### Changing the threshold

| Threshold | Accuracy | Precision | Recall |
|---|---|---|---|
| 0.3 | 0.850 | 0.788 | 0.929 |
| 0.4 | 0.850 | 0.828 | 0.857 |
| 0.5 (default) | 0.817 | 0.870 | 0.714 |
| 0.6 | 0.817 | 0.870 | 0.714 |
| 0.7 | 0.800 | 0.900 | 0.643 |

For a medical problem, missing a sick patient is worse than a false alarm, so recall matters most. Lowering the threshold from 0.5 to 0.3 raises recall from 0.714 to 0.929 (26 of 28 sick patients found instead of 20), at the cost of more false alarms (3 to 7) and lower precision.

### What drives the prediction

Strongest signals raising the predicted risk: asymptomatic chest pain, a reversible thalassemia defect, 1 or 2 flagged blood vessels, higher ST depression (`oldpeak`), and being male. Strongest signals lowering it: zero flagged vessels (the largest coefficient overall), non-anginal pain, a normal thalassemia result, and an upsloping ST slope. These are patterns the model learned from a small dataset, not proof of cause and effect.

## Limitations

- Small dataset: the test set has only 60 patients, 28 of them sick, so one missed patient moves recall by about 3.6 points. Results could change noticeably with a different split.
- All patients come from one hospital and were already being tested for heart problems, so the model may not generalize to the general population.
- The thresholds in the table above were compared on the test set, so they illustrate the trade-off rather than a final tuned choice. A stricter approach would choose the threshold with cross-validated predictions on the training set.
- Six patients with missing values were dropped (about 2%).

## Next steps

- Compare with K-Nearest Neighbors and Random Forest on the same data.
- Report cross-validated recall and ROC-AUC on the whole dataset for steadier estimates.
- Choose the threshold using training-set cross-validation.
- Try a larger heart-disease dataset.

## Files

| File | Purpose |
|---|---|
| `heart_disease_prediction.ipynb` | Full analysis (open in Google Colab) |
| `model.json` | Trained coefficients, scaling values and categories |
| `index.html` | Web page that recomputes the prediction in JavaScript from `model.json` |

## How the website works

Logistic Regression is arithmetic: scale the numeric inputs, add the one-hot columns, multiply by the coefficients, add the intercept, and pass the sum through the sigmoid function. `model.json` stores those numbers and `index.html` repeats the steps in the browser, so no server is needed. For the test patient used to verify the page (age 41, male, atypical angina, no flagged vessels, normal thalassemia, and so on), both the notebook and the website give **2.5%**.

## Tools

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, HTML/JavaScript, GitHub Pages.
