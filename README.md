# Bank Marketing ML Pipeline with Spark

This project implements a complete machine learning workflow for the UCI Bank Marketing dataset using Apache Spark and PySpark. It covers data ingestion from HDFS, feature engineering, model training (Logistic Regression, Decision Tree, Random Forest), hyperparameter tuning, evaluation, and visualization of results (ROC curve, confusion matrix).

## Repository Structure

* `bank_marketing_ml.py` — Main PySpark script that runs the end-to-end pipeline
* `data/` — Location in HDFS for input dataset (download and put under `hdfs:///user/you/bank/`)
* `output/` — Location in HDFS for output metrics and plot files
* `README.md` — This file

## Requirements

* Apache Spark 2.4+ with PySpark (We used spark 3.5.5)
* Hadoop HDFS
* Python 3.8+
* Python libraries:

  * `pandas`
  * `matplotlib`
  * `scikit-learn`
  * `numpy`
  * `scipy`

## Setup & Usage

1. **We had to install Python 3.8 (was not easy on Ubuntu 16.04)**

   ```bash
   export PYSPARK_PYTHON=/usr/bin/python3.8
   export PYSPARK_DRIVER_PYTHON=/usr/bin/python3.8
   ```

2. **Place dataset in HDFS**

   ```bash
   hadoop dfs -mkdir -p /user/hadoopuser/bank
   hadoop dfs -put bank-additional-full.csv /user/hadoopuser/bank/
   ```

3. **Run the pipeline**

   ```bash
   spark-submit \
     --master local[*] \
     bank_marketing_ml.py \
     hdfs:///user/hadoopuser/bank/bank-additional-full.csv \
     hdfs:///user/hadoopuser/bank/output
   ```

4. **Retrieve output plots**

   ```bash
   hadoop dfs -get /user/hadoopuser/bank/output/roc_curve.png  .
   hadoop dfs -get /user/hadoopuser/bank/output/confusion_matrix.png  .
   ```

## Results

* **AUC & Accuracy** printed in console logs
* **`roc_curve.png`** and **`confusion_matrix.png`** saved under the specified output path in HDFS

## License

This project is released under the MIT License.
