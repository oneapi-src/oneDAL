/* file: LinRegModelBuilder.java */
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
 //     Java example of linear regression model builder.
 //
 //     The program trains the linear regression model on a training data set with
 //     the normal equations method and computes regression for the test data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-LINEARREGRESSIONMODELBUILDER">
 * @example LinRegModelBuilder.java
 */

package com.intel.daal.examples.linear_regression;

import java.nio.FloatBuffer;
import com.intel.daal.algorithms.linear_regression.Model;
import com.intel.daal.algorithms.linear_regression.ModelBuilder;
import com.intel.daal.algorithms.linear_regression.prediction.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class LinRegModelBuilder {
    /* Input data set parameters */
    private static final String trainDatasetFileName = "../data/batch/linear_regression_trained_model.csv";
    private static final String testDatasetFileName  = "../data/batch/linear_regression_test.csv";

    private static final int nFeatures            = 10;  /* Number of features in training and testing data sets */
    private static final int nDependentVariables  = 2;   /* Number of dependent variables that correspond to each observation */

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
        /* Get beta from trained model */
        trainDataSource.loadDataBlock(betaData);

        /* Define the size of beta */
        long nBeta = betaData.getNumberOfRows()*betaData.getNumberOfColumns();

        /* Initialize beta buffer */
        FloatBuffer bufferBeta = FloatBuffer.allocate(0);
        bufferBeta = betaData.getBlockOfRows(0, betaData.getNumberOfRows(), bufferBeta);

        /*Convert from buffer to array */
        float [] arrayBeta = new float[(int)nBeta];
        bufferBeta.position(0);
        bufferBeta.get(arrayBeta);

        /* Create model builder */
        ModelBuilder modelBuilder = new ModelBuilder(context, Float.class, nFeatures, nDependentVariables);

        /* Set beta */
        modelBuilder.setBeta(arrayBeta);

        model = modelBuilder.getModel();
    }

    private static void testModel() {
        /* Initialize FileDataSource to retrieve the test data from a .csv file */
        FileDataSource testDataSource = new FileDataSource(context, testDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for testing data and ground truth labels */
        NumericTable testData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
        testGroundTruth = new HomogenNumericTable(context, Float.class, nDependentVariables, 0, NumericTable.AllocationFlag.DoNotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(testData);
        mergedData.addNumericTable(testGroundTruth);

        /* Load the data from the input file */
        testDataSource.loadDataBlock(mergedData);

        /* Create an algorithm object to predict values of multiple linear regression */
        PredictionBatch linearRegressionPredict = new PredictionBatch(context, Float.class,
                PredictionMethod.defaultDense);

        linearRegressionPredict.input.set(PredictionInputId.data, testData);
        linearRegressionPredict.input.set(PredictionInputId.model, model);

        /* Compute prediction results */
        PredictionResult predictionResult = linearRegressionPredict.compute();

        results = predictionResult.get(PredictionResultId.prediction);
    }

    private static void printResults() {
        Service.printNumericTable("Linear Regression coefficients of built model:", model.getBeta());
        Service.printNumericTable("Linear Regression prediction results: (first 10 rows):", results, 10);
        Service.printNumericTable("Ground truth (first 10 rows):", testGroundTruth, 10);

    }
}
