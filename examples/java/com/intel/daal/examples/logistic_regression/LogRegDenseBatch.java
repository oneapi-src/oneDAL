/* file: LogRegDenseBatch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
 //     Java example of logistic regression in the batch processing mode.
 //
 //     The program trains the logistic regression model on a training data set with
 //     the normal equations method and computes regression for the test data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-LOGISTICREGRESSIONDENSEBATCH">
 * @example LogRegDenseBatch.java
 */

package com.intel.daal.examples.logistic_regression;

import com.intel.daal.algorithms.logistic_regression.Model;
import com.intel.daal.algorithms.logistic_regression.prediction.*;
import com.intel.daal.algorithms.logistic_regression.training.*;
import com.intel.daal.algorithms.classifier.training.InputId;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.classifier.prediction.ModelInputId;
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId;
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class LogRegDenseBatch {
    /* Input data set parameters */
    private static final String trainDatasetFileName = "../data/batch/logreg_train.csv";

    private static final String testDatasetFileName  = "../data/batch/logreg_test.csv";

    private static final int nFeatures            = 6;  /* Number of features in training and testing data sets */
    private static final int nClasses             = 5;  /* Number of classes */

    static Model        model;
    static NumericTable results;
    static NumericTable probabilities;
    static NumericTable logProbabilities;
    static NumericTable testDependentVariables;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        trainModel();

        testModel();

        printResults();

        context.dispose();
    }

    private static void trainModel() {

        /* Initialize FileDataSource to retrieve the input data from a .csv file */
        FileDataSource trainDataSource = new FileDataSource(context, trainDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for training data and labels */
        NumericTable trainData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
        NumericTable trainDependentVariables = new HomogenNumericTable(context, Float.class, 1, 0,
                                                                       NumericTable.AllocationFlag.DoNotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(trainData);
        mergedData.addNumericTable(trainDependentVariables);

        /* Retrieve the data from an input file */
        trainDataSource.loadDataBlock(mergedData);

        /* Create an algorithm object to train the multiple logistic regression model */
        TrainingBatch logisticRegressionTrain = new TrainingBatch(context, Float.class, TrainingMethod.defaultDense, nClasses);

        logisticRegressionTrain.input.set(InputId.data, trainData);
        logisticRegressionTrain.input.set(InputId.labels, trainDependentVariables);
        logisticRegressionTrain.parameter.setPenaltyL1(0.1f);
        logisticRegressionTrain.parameter.setPenaltyL2(0.1f);

        /* Build the multiple logistic regression model */
        TrainingResult trainingResult = logisticRegressionTrain.compute();

        model = trainingResult.get(TrainingResultId.model);
    }

    private static void testModel() {
        /* Initialize FileDataSource to retrieve the input data from a .csv file */
        FileDataSource testDataSource = new FileDataSource(context, testDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for testing data and labels */
        NumericTable testData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
        testDependentVariables = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.DoNotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(testData);
        mergedData.addNumericTable(testDependentVariables);

        /* Retrieve the data from an input file */
        testDataSource.loadDataBlock(mergedData);

        /* Create algorithm objects to predict values of multiple logistic regression with the default method */
        PredictionBatch logisticRegressionPredict = new PredictionBatch(context, Float.class,
                PredictionMethod.defaultDense, nClasses);

        logisticRegressionPredict.input.set(NumericTableInputId.data, testData);
        logisticRegressionPredict.input.set(ModelInputId.model, model);

        logisticRegressionPredict.parameter.setResultsToCompute(PredictionResultsToComputeId.computeClassesLabels|PredictionResultsToComputeId.computeClassesProbabilities|PredictionResultsToComputeId.computeClassesLogProbabilities);

        /* Compute prediction results */
        PredictionResult predictionResult = logisticRegressionPredict.compute();

        results = predictionResult.get(PredictionResultId.prediction);
        probabilities = predictionResult.get(PredictionResultNumericTableId.probabilities);
        logProbabilities = predictionResult.get(PredictionResultNumericTableId.logProbabilities);
    }

    private static void printResults() {
        NumericTable beta = model.getBeta();
        NumericTable expected = testDependentVariables;
        Service.printNumericTable("Logistic Regression coefficients: ", beta);
        Service.printNumericTable("Logistic Regression prediction results: (first 10 rows):", results, 10);
        Service.printNumericTable("Ground truth (first 10 rows):", expected, 10);
        Service.printNumericTable("Logistic Regression prediction probabilities: (first 10 rows):", probabilities, 10);
        Service.printNumericTable("Logistic Regression prediction log probabilities: (first 10 rows):", logProbabilities, 10);
    }
}
