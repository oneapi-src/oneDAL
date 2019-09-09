/* file: SVMMultiClassMetricsDenseBatch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
 //  Content:
 //     Java example of multi-class support vector machine (SVM) quality metrics
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-SVMMULTICLASSQUALITYMETRICSETBATCHEXAMPLE">
 * @example SVMMultiClassMetricsDenseBatch.java
 */

package com.intel.daal.examples.quality_metrics;

import java.nio.DoubleBuffer;

import com.intel.daal.algorithms.classifier.prediction.ModelInputId;
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId;
import com.intel.daal.algorithms.classifier.prediction.PredictionResult;
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId;
import com.intel.daal.algorithms.classifier.quality_metric.multi_class_confusion_matrix.*;
import com.intel.daal.algorithms.classifier.training.InputId;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.multi_class_classifier.Model;
import com.intel.daal.algorithms.multi_class_classifier.prediction.*;
import com.intel.daal.algorithms.multi_class_classifier.quality_metric_set.*;
import com.intel.daal.algorithms.multi_class_classifier.training.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class SVMMultiClassMetricsDenseBatch {

    /* Input data set parameters */
    private static final String trainDatasetFileName     = "../data/batch/svm_multi_class_train_dense.csv";

    private static final String testDatasetFileName     = "../data/batch/svm_multi_class_test_dense.csv";

    private static final int nFeatures     = 20;
    private static final int nClasses      = 5;

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
        com.intel.daal.algorithms.svm.training.TrainingBatch training = new com.intel.daal.algorithms.svm.training.TrainingBatch(
                context, Float.class, com.intel.daal.algorithms.svm.training.TrainingMethod.boser);

        com.intel.daal.algorithms.svm.prediction.PredictionBatch prediction = new com.intel.daal.algorithms.svm.prediction.PredictionBatch(
                context, Float.class, com.intel.daal.algorithms.svm.prediction.PredictionMethod.defaultDense);

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

        /* Create algorithm objects to train the multi-class SVM model */
        TrainingBatch algorithm = new TrainingBatch(context, Float.class, TrainingMethod.oneAgainstOne, nClasses);

        /* Set parameters for the multi-class SVM algorithm */
        algorithm.parameter.setTraining(training);
        algorithm.parameter.setPrediction(prediction);

        /* Pass a training data set and dependent values to the algorithm */
        algorithm.input.set(InputId.data, trainData);
        algorithm.input.set(InputId.labels, trainGroundTruth);

        /* Train the multi-class SVM model */
        trainingResult = algorithm.compute();
    }

    private static void testModel() {
        com.intel.daal.algorithms.svm.training.TrainingBatch training = new com.intel.daal.algorithms.svm.training.TrainingBatch(
                context, Float.class, com.intel.daal.algorithms.svm.training.TrainingMethod.boser);

        com.intel.daal.algorithms.svm.prediction.PredictionBatch prediction = new com.intel.daal.algorithms.svm.prediction.PredictionBatch(
                context, Float.class, com.intel.daal.algorithms.svm.prediction.PredictionMethod.defaultDense);

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

        /* Create algorithm objects to predict multi-class SVM values with the defaultDense method */
        PredictionBatch algorithm = new PredictionBatch(context, Float.class, PredictionMethod.multiClassClassifierWu, nClasses);

        algorithm.parameter.setTraining(training);
        algorithm.parameter.setPrediction(prediction);

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
        QualityMetricSetBatch quality_metric_set = new QualityMetricSetBatch(context, nClasses);

        MultiClassConfusionMatrixInput input = quality_metric_set.getInputDataCollection()
                .getInput(QualityMetricId.confusionMatrix);

        input.set(MultiClassConfusionMatrixInputId.predictedLabels, predictedLabels);
        input.set(MultiClassConfusionMatrixInputId.groundTruthLabels, groundTruthLabels);

        /* Compute quality metrics */
        qualityMetricSetResult = quality_metric_set.compute();
    }

    private static void printResults() {
        /* Print the classification results */
        Service.printClassificationResult(groundTruthLabels, predictedLabels, "Ground truth", "Classification results",
                "SVM classification results (first 20 observations):", 20);
        /* Print the quality metrics */
        MultiClassConfusionMatrixResult qualityMetricResult = qualityMetricSetResult
                .getResult(QualityMetricId.confusionMatrix);
        NumericTable confusionMatrix = qualityMetricResult.get(MultiClassConfusionMatrixResultId.confusionMatrix);
        NumericTable multiClassMetrics = qualityMetricResult.get(MultiClassConfusionMatrixResultId.multiClassMetrics);

        Service.printNumericTable("\nConfusion matrix:", confusionMatrix);

        DoubleBuffer qualityMetricsData = DoubleBuffer
                .allocate((int) (multiClassMetrics.getNumberOfColumns() * multiClassMetrics.getNumberOfRows()));
        qualityMetricsData = multiClassMetrics.getBlockOfRows(0, multiClassMetrics.getNumberOfRows(),
                qualityMetricsData);

        System.out
                .println("Average accuracy: " + qualityMetricsData.get(MultiClassMetricId.averageAccuracy.getValue()));
        System.out.println("Error rate:       " + qualityMetricsData.get(MultiClassMetricId.errorRate.getValue()));
        System.out.println("Micro precision:  " + qualityMetricsData.get(MultiClassMetricId.microPrecision.getValue()));
        System.out.println("Micro recall:     " + qualityMetricsData.get(MultiClassMetricId.microRecall.getValue()));
        System.out.println("Micro F-score:    " + qualityMetricsData.get(MultiClassMetricId.microFscore.getValue()));
        System.out.println("Macro precision:  " + qualityMetricsData.get(MultiClassMetricId.macroPrecision.getValue()));
        System.out.println("Macro recall:     " + qualityMetricsData.get(MultiClassMetricId.macroRecall.getValue()));
        System.out.println("Macro F-score:    " + qualityMetricsData.get(MultiClassMetricId.macroFscore.getValue()));
    }
}
