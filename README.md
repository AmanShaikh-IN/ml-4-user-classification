# Machine Learning for User Classification

## Case Description

### Background
In a machine learning classification problem, the algorithm assigns labels to instances based on their features. This *Machine Learning for User Classification* project involves utilizing a subset of our internal data, stripped of personally identifiable information, to predict user behavior. Specifically, you will analyze student engagement metrics, such as:
- The number of days students have spent on the platform.
- The total minutes of watched content.
- The number of courses students have started.

### Goal
The objective is to predict whether students will upgrade from a free plan to a paid plan, using the provided dataset and the models above.

### Business Objective
The ability to predict potential paying customers is critical not only for 365 Data Science but for any online platform. Such predictions can:
- Enhance targeted advertising campaigns.
- Support outreach initiatives with exclusive offers.
- Optimize marketing budgets by focusing on users most likely to convert to paid customers.

This project aims to increase company revenue through data-driven strategies.

---

**Important Note:**  
The dataset is heavily imbalanced, with significantly fewer students likely to upgrade to a paid plan. While addressing class imbalance with methods such as oversampling, undersampling, or SMOTE is optional for this project, it is encouraged to improve model performance.

---

## Project Requirements
This project requires Python 3 (or newer) and the following libraries:

- **pandas**: For data manipulation.
- **matplotlib**: For basic visualizations.
- **statsmodels**: For statistical analysis.
- **scikit-learn**: For machine learning models.
- **numpy**: For numerical computations.
- **seaborn**: For advanced data visualizations.

To install the libraries, you can use the following command:
```bash
pip install pandas matplotlib statsmodels scikit-learn numpy seaborn

