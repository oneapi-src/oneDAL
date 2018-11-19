/* file: SVMTwoClassMetricsDenseBatch.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
 //  Content:
 //     Java example of two-class support vector machine (SVM) quality metrics
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-SVMTWOCLASSQUALITYMETRICSETBATCHEXAMPLE">
 * @example SVMTwoClassMetricsDenseBatch.java
 */

package com.intel.daal.examples.quality_metrics;

import java.nio.DoubleBuffer;

import com.intel.daal.algorithms.classifier.prediction.ModelInputId;
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId;
import com.intel.daal.algorithms.classifier.prediction.PredictionResult;
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId;
import com.intel.daal.algorithms.classifier.quality_metric.binary_confusion_matrix.*;
import com.intel.daal.algorithms.classifier.training.InputId;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.svm.Model;
import com.intel.daal.algorithms.svm.prediction.PredictionBatch;
import com.intel.daal.algorithms.svm.prediction.PredictionMethod;
import com.intel.daal.algorithms.svm.quality_metric_set.*;
import com.intel.daal.algorithms.svm.training.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class SVMTwoClassMetricsDenseBatch {

    /* Input data set parameters */
    private static final String trainDatasetFileName     = "../data/batch/svm_two_class_train_dense.csv";

    private static final String testDatasetFileName     = "../data/batch/svm_two_class_test_dense.csv";

    private static final int nFeatures     = 20;

    private static TrainingResult   trainingResult;
    private static PredictionResult predictionResult;
    private static ResultCollection qualityMetricSetResult;

    private static NumericTable groundTruthLabels;
    private static NumericTable predictedLabels;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        trainModel();

        testModel();

        testModelQuality();

        printResults();

        context.dispose();
    }

    private static void trainModel() {
        /* Retrieve the data from input data sets */
        FileDataSource trainDataSource = new FileDataSource(context, trainDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for training data and labels */
        NumericTable trainData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
        NumericTable trainGroundTruth = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.DoNotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(trainData);
        mergedData.addNumericTable(trainGroundTruth);

        /* Retrieve the data from an input file */
        trainDataSource.loadDataBlock(mergedData);

        /* Create algorithm objects to train the two-class SVM model */
        TrainingBatch algorithm = new TrainingBatch(context, Float.class, TrainingMethod.boser);

        /* Set parameters for the two-class SVM algorithm */
        algorithm.parameter.setCacheSize(40000000);
        algorithm.parameter
                .setKernel(new com.intel.daal.algorithms.kernel_function.linear.Batch(context, Float.class));

        /* Pass a training data set and dependent values to the algorithm */
        algorithm.input.set(InputId.data, trainData);
        algorithm.input.set(InputId.labels, trainGroundTruth);

        /* Train the two-class SVM model */
        trainingResult = algorithm.compute();
    }

    private static void testModel() {
        FileDataSource testDataSource = new FileDataSource(context, testDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for testing data and labels */
        NumericTable testData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
        groundTruthLabels = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.DoNotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(testData);
        mergedData.addNumericTable(groundTruthLabels);

        /* Retrieve the data from an input file */
        testDataSource.loadDataBlock(mergedData);

        /* Create algorithm objects to predict two-class SVM values with the defaultDense method */
        PredictionBatch algorithm = new PredictionBatch(context, Float.class, PredictionMethod.defaultDense);

        algorithm.parameter
                .setKernel(new com.intel.daal.algorithms.kernel_function.linear.Batch(context, Float.class));

        Model model = trainingResult.get(TrainingResultId.model);

        /* Pass a testing data set and the trained model to the algorithm */
        algorithm.input.set(NumericTableInputId.data, testData);
        algorithm.input.set(ModelInputId.model, model);

        /* Compute the prediction results */
        predictionResult = algorithm.compute();
    }

    private static void testModelQuality() {
        /* Retrieve predicted labels */
        predictedLabels = predictionResult.get(PredictionResultId.prediction);

        /* Create a quality metric set object to compute quality metrics of the SVM algorithm */
        QualityMetricSetBatch quality_metric_set = new QualityMetricSetBatch(context);

        BinaryConfusionMatrixInput input = quality_metric_set.getInputDataCollection()
                .getInput(QualityMetricId.confusionMatrix);

        input.set(BinaryConfusionMatrixInputId.predictedLabels, predictedLabels);
        input.set(BinaryConfusionMatrixInputId.groundTruthLabels, groundTruthLabels);

        /* Compute quality metrics */
        qualityMetricSetResult = quality_metric_set.compute();
    }

    private static void printResults() {
        /* Print the classification results */
        Service.printClassificationResult(groundTruthLabels, predictedLabels, "Ground truth", "Classification results",
                "SVM classification results (first 20 observations):", 20);
        /* Print the quality metrics */
        BinaryConfusionMatrixResult qualityMetricResult = qualityMetricSetResult
                .getResult(QualityMetricId.confusionMatrix);
        NumericTable confusionMatrix = qualityMetricResult.get(BinaryConfusionMatrixResultId.confusionMatrix);
        NumericTable binaryMetrics = qualityMetricResult.get(BinaryConfusionMatrixResultId.binaryMetrics);

        Service.printNumericTable("\nConfusion matrix:", confusionMatrix);

        DoubleBuffer qualityMetricsData = DoubleBuffer
                .allocate((int) (binaryMetrics.getNumberOfColumns() * binaryMetrics.getNumberOfRows()));
        qualityMetricsData = binaryMetrics.getBlockOfRows(0, binaryMetrics.getNumberOfRows(), qualityMetricsData);

        System.out.println("Accuracy:      " + qualityMetricsData.get(BinaryMetricId.accuracy.getValue()));
        System.out.println("Precision:     " + qualityMetricsData.get(BinaryMetricId.precision.getValue()));
        System.out.println("Recall:        " + qualityMetricsData.get(BinaryMetricId.recall.getValue()));
        System.out.println("F-score:       " + qualityMetricsData.get(BinaryMetricId.fscore.getValue()));
        System.out.println("Specificity:   " + qualityMetricsData.get(BinaryMetricId.specificity.getValue()));
        System.out.println("AUC:           " + qualityMetricsData.get(BinaryMetricId.AUC.getValue()));
    }
}
