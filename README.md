# PGP-AIML Projects

End-to-end Machine Learning and Deep Learning projects completed as part of a
PG Program in AI & ML. Each project includes a full Jupyter notebook (EDA,
modeling, evaluation, and business recommendations) along with a synthetic
dataset that mirrors the real data's schema and statistics, so every notebook
can be run and reviewed without exposing any real, proprietary, or
confidential data.

## Projects

| # | Project | Domain | Techniques |
|---|---|---|---|
| 01 | [FoodHub Data Analysis](./01-foodhub-eda) | Food delivery / EDA | pandas, seaborn, exploratory data analysis |
| 02 | [ReneWind](./02-renewind-neural-network) | Predictive maintenance | Neural networks (Keras/TensorFlow), class imbalance handling |
| 03 | [HelmNet](./03-helmnet-cnn-classification) | Workplace safety / Computer vision | CNNs, transfer learning (VGG16), data augmentation |
| 04 | [EasyVisa](./04-easyvisa-classification) | Immigration / Classification | Ensemble methods, SMOTE, hyperparameter tuning |

---

### 01. FoodHub Data Analysis

Exploratory data analysis on a food-delivery order dataset (1,898 orders)
using pandas, seaborn, and matplotlib to uncover customer ordering patterns,
restaurant performance, and delivery-time trends, with business
recommendations for a food-delivery platform.

**Techniques:** univariate & multivariate analysis, groupby aggregation,
correlation analysis, data visualization

**Data:** synthetic dataset matching the real data's schema and statistics
(see `generate_dummy_foodhub.py`)

---

### 02. ReneWind — Wind Turbine Failure Prediction

Binary classification on imbalanced sensor data to predict wind-turbine
generator failures before they happen. Compares 7 neural network
architectures (Keras/TensorFlow) with class weighting, evaluating trade-offs
between accuracy and recall in a cost-sensitive predictive-maintenance
scenario.

**Techniques:** neural networks, class weighting, model comparison,
threshold analysis

**Data:** synthetic dataset matching the real data's schema, size, and class
imbalance (see `generate_dummy_renewind.py`)

---

### 03. HelmNet — Safety Helmet Detection

Image classification system to detect whether workers are wearing safety
helmets on construction/industrial sites. Compares a custom CNN trained from
scratch against VGG16 transfer-learning variants with data augmentation.

**Techniques:** CNNs, transfer learning, data augmentation, model comparison

**Data:** synthetic placeholder images (not real workplace photos) matching
the real dataset's size and class balance (see `generate_dummy_helmnet.py`)

---

### 04. EasyVisa — US Visa Application Classification

Binary classification predicting whether a US work visa application will be
certified or denied, based on applicant, employer, and wage attributes.
Compares 6 ensemble classifiers across original, oversampled (SMOTE), and
undersampled training data, with hyperparameter tuning via
RandomizedSearchCV.

**Techniques:** ensemble methods (Bagging, Random Forest, GBM, AdaBoost,
XGBoost, Decision Tree), SMOTE/undersampling, RandomizedSearchCV,
feature importance analysis

**Data:** synthetic dataset matching the real data's schema and statistics
(see `generate_dummy_easyvisa.py`)

---

## A note on the data

None of the datasets in this repository contain real customer, employee,
patient, or business data. Each project includes a `generate_dummy_*.py`
script that produces a synthetic dataset matching the schema, size, and key
statistics (class balance, value ranges, correlations) of the original real
dataset used during coursework, so the notebooks can be run and reviewed by
anyone without any privacy or confidentiality concerns.

## Repo Structure

```
pgp-aiml-projects/
├── README.md
├── 01-foodhub-eda/
│   ├── FoodHub_Data_Analysis.ipynb
│   ├── generate_dummy_foodhub.py
│   └── foodhub_order_DUMMY.csv
├── 02-renewind-neural-network/
│   ├── ReneWind_Neural_Network.ipynb
│   ├── generate_dummy_renewind.py
│   ├── Train_DUMMY.csv
│   └── Test_DUMMY.csv
├── 03-helmnet-cnn-classification/
│   ├── HelmNet_CNN_Classification.ipynb
│   ├── generate_dummy_helmnet.py
│   ├── images_proj_DUMMY.npy
│   └── Labels_proj_DUMMY.csv
└── 04-easyvisa-classification/
    ├── EasyVisa_Classification.ipynb
    ├── generate_dummy_easyvisa.py
    └── EasyVisa_DUMMY.csv
```

## Tools & Libraries

Python · pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn ·
TensorFlow / Keras · seaborn · matplotlib

## License

Code and synthetic datasets in this repository are original work / generated
data and are provided as-is for portfolio and educational purposes.
