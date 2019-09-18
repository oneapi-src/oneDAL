/* file: LogRegModelBuilder.java */
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
 //     Java example of logistic regression model builder.
 //
 //     The program trains the logistic regression model on a training data set with
 //     the normal equations method and computes regression for the test data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-LOGISTICREGRESSIONMODELBUILDER">
 * @example LogRegModelBuilder.java
 */

package com.intel.daal.examples.logistic_regression;

import java.nio.FloatBuffer;
import com.intel.daal.algorithms.logistic_regression.Model;
import com.intel.daal.algorithms.logistic_regression.ModelBuilder;
import com.intel.daal.algorithms.logistic_regression.prediction.*;
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

class LogRegModelBuilder {
    /* Input data set parameters */
    private static final String trainDatasetFileName = "../data/batch/logreg_trained_model.csv";
    private static final String testDatasetFileName  = "../data/batch/logreg_test.csv";

    private static final int nFeatures            = 6;  /* Number of features in training and testing data sets */
    private static final int nClasses             = 5;  /* Number of classes */

    static Model        model;
    static NumericTable results;
    static NumericTable testGroundTruth;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        buildModel();
        testModel();
        printResults();
        context.dispose();
    }

    public static void buildModel() {
        /* Initialize FileDataSource to retrieve the beta data from a .csv file */
        FileDataSource trainDataSource = new FileDataSource(context, trainDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

        /* Create Numeric Table for beta coefficients */
        NumericTable betaData = new HomogenNumericTable(context, Float.class, nFeatures + 1, 0, NumericTable.AllocationFlag.DoNotAllocate);
        trainDataSource.loadDataBlock(betaData);

        /* Define the size of beta */
        long nBeta = betaData.getNumberOfRows()*betaData.getNumberOfColumns();

        /* Initialize beta array */
        FloatBuffer bufferBeta = FloatBuffer.allocate(0);
        bufferBeta = betaData.getBlockOfRows(0, nClasses, bufferBeta);

        /* Create model builder */
        ModelBuilder modelBuilder = new ModelBuilder(context, Float.class, nFeatures, nClasses);

        /* Set beta */
        modelBuilder.setBeta(bufferBeta, nBeta);

        model = modelBuilder.getModel();
    }

    private static void testModel() {
        /* Initialize FileDataSource to retrieve the test data from a .csv file */
        FileDataSource testDataSource = new FileDataSource(context, testDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for testing data and ground truth labels */
        NumericTable testData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
        testGroundTruth = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.DoNotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(testData);
        mergedData.addNumericTable(testGroundTruth);

        /* Load the data from the input file */
        testDataSource.loadDataBlock(mergedData);

        /* Create algorithm objects to predict values of multiple logistic regression with the default method */
        PredictionBatch logisticRegressionPredict = new PredictionBatch(context, Float.class,
                PredictionMethod.defaultDense, nClasses);

        logisticRegressionPredict.input.set(NumericTableInputId.data, testData);
        logisticRegressionPredict.input.set(ModelInputId.model, model);

        logisticRegressionPredict.parameter.setResultsToEvaluate(ResultsToComputeId.computeClassLabels);

        /* Compute prediction results */
        PredictionResult predictionResult = logisticRegressionPredict.compute();

        results = predictionResult.get(PredictionResultId.prediction);
    }

    private static void printResults() {
        Service.printNumericTable("Logistic Regression coefficients of built model:", model.getBeta());
        Service.printNumericTable("Logistic Regression prediction results: (first 10 rows):", results, 10);
        Service.printNumericTable("Ground truth (first 10 rows):", testGroundTruth, 10);

    }
}
