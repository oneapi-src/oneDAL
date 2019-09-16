/* file: LogRegBinaryDenseBatch.java */
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
 //     Java example of logistic regression in the batch processing mode.
 //
 //     The program trains the logistic regression model on a training data set with
 //     the normal equations method and computes regression for the test data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-LOGISTICREGRESSIONBINARYDENSEBATCH">
 * @example LogRegBinaryDenseBatch.java
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
import com.intel.daal.algorithms.classifier.prediction.PredictionResult;
import com.intel.daal.algorithms.classifier.ResultsToComputeId;
import com.intel.daal.algorithms.classifier.Parameter;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class LogRegBinaryDenseBatch {
    /* Input data set parameters */
    private static final String trainDatasetFileName = "../data/batch/binary_cls_train.csv";

    private static final String testDatasetFileName  = "../data/batch/binary_cls_test.csv";

    private static final int nFeatures            = 20;  /* Number of features in training and testing data sets */
    private static final int nClasses             = 2;   /* Number of classes */

    static Model        model;
    static NumericTable results;
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

        /* Create an algorithm object to train the multiple logistic regression model with the normal equations method */
        TrainingBatch logisticRegressionTrain = new TrainingBatch(context, Float.class, TrainingMethod.defaultDense, nClasses);

        logisticRegressionTrain.input.set(InputId.data, trainData);
        logisticRegressionTrain.input.set(InputId.labels, trainDependentVariables);

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

        /* Compute prediction results */
        PredictionResult predictionResult = logisticRegressionPredict.compute();

        results = predictionResult.get(PredictionResultId.prediction);
    }

    private static void printResults() {
        NumericTable beta = model.getBeta();
        NumericTable expected = testDependentVariables;
        Service.printNumericTable("Logistic Regression coefficients: ", beta);
        Service.printNumericTable("Logistic Regression prediction results: (first 10 rows):", results, 10);
        Service.printNumericTable("Ground truth (first 10 rows):", expected, 10);
    }
}
