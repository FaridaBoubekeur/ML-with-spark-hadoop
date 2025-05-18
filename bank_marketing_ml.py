#!/usr/bin/env python3

import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
 
import matplotlib.pyplot as plt
import numpy as np
from pyspark.mllib.evaluation import BinaryClassificationMetrics

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import subprocess


def main(hdfs_input, hdfs_output):
    spark = SparkSession.builder.appName("BankMarketingML").getOrCreate()

    # 1) Load data
    df = spark.read.csv(hdfs_input, header=True, sep=";", inferSchema=True)

    # 1.1) Rename columns containing dots
    for old in ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "nr.employed"]:
        df = df.withColumnRenamed(old, old.replace(".", "_"))

    # 2) Feature engineering
    catCols = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "poutcome",
    ]
    indexers = [
        StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
        for c in catCols
    ]
    encoders = [
        OneHotEncoder(inputCol=c + "_idx", outputCol=c + "_vec") for c in catCols
    ]
    numericCols = [
        "age",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "emp_var_rate",
        "cons_price_idx",
        "cons_conf_idx",
        "euribor3m",
        "nr_employed",
    ]
    assembler = VectorAssembler(
        inputCols=[c + "_vec" for c in catCols] + numericCols, outputCol="features"
    )
    labelIndexer = StringIndexer(inputCol="y", outputCol="label")

    # 3) Split into train/test
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    # 4) Define models + evaluator
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
    rf = RandomForestClassifier(featuresCol="features", labelCol="label")
    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

    # 5) Param grid (example tuning LR and RF)
    paramGrid = (
        ParamGridBuilder()
        .addGrid(lr.maxIter, [10, 50])
        .addGrid(rf.numTrees, [20, 50])
        .build()
    )

    # 6) Cross-validator
    cv = CrossValidator(
        estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3
    )

    # 7) Build full pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler, labelIndexer, cv])

    # 8) Fit the pipeline to training data
    model = pipeline.fit(train)

    # 9) Make predictions & evaluate
    preds = model.transform(test)
    aucf = evaluator.evaluate(preds)
    acc = preds.filter("label = prediction").count() / test.count()

    print(f"=== Final metrics on test set ===")
    print(f"AUC      = {aucf:.4f}")
    print(f"Accuracy = {acc:.4f}")

    # 10) Detailed classification metrics via MulticlassMetrics
    # … right after you print Precision/Recall/F1 …

        # --- PLOT & SAVE section ---

    # 3.1) pull into Pandas
    pdf = preds.select("probability", "label") \
               .toPandas()
    y_true  = pdf["label"]
    y_score = pdf["probability"].apply(lambda v: v[1])

    # 3.2) ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0,1],[0,1], lw=1, linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    local_roc = "/tmp/roc_curve.png"
    plt.savefig(local_roc)
    plt.close()

    # push to HDFS
    subprocess.check_call(["hdfs", "dfs", "-mkdir", "-p", hdfs_output])
    subprocess.check_call(["hdfs", "dfs", "-put", "-f", local_roc, hdfs_output])

    # 3.3) Confusion matrix
    cm = confusion_matrix(y_true, (y_score > 0.5).astype(int))
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks([0,1], ["no","yes"])
    plt.yticks([0,1], ["no","yes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    local_cm = "/tmp/confusion_matrix.png"
    plt.savefig(local_cm)
    plt.close()

    # push to HDFS
    subprocess.check_call(["hdfs", "dfs", "-put", "-f", local_cm, hdfs_output])


    # now stop Spark
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: bank_marketing_ml.py <hdfs_input> <hdfs_output>")
        sys.exit(-1)
    main(sys.argv[1], sys.argv[2])
