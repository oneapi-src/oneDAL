/* file: LinRegMetricsDenseBatch.java */
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
 //     Java example of multiple linear regression in the batch processing mode.
 //
 //     The program trains the multiple linear regression model on a training
 //     data set with the normal equations method and computes regression for
 //     the test data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-LINREGQUALITYMETRICSETBATCHEXAMPLE">
 * @example LinRegMetricsDenseBatch.java
 */

package com.intel.daal.examples.quality_metrics;

import java.nio.DoubleBuffer;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.linear_regression.Model;
import com.intel.daal.algorithms.linear_regression.prediction.*;
import com.intel.daal.algorithms.linear_regression.training.*;
import com.intel.daal.algorithms.linear_regression.quality_metric.*;
import com.intel.daal.algorithms.linear_regression.quality_metric_set.*;
import com.intel.daal.data_management.data.DataCollection;

class LinRegMetricsDenseBatch {
    /* Input data set parameters */
    private static final String trainDatasetFileName = "../data/batch/linear_regression_train.csv";

    private static final int nFeatures           = 10;
    private static final int nDependentVariables = 2;
    private static final int iBeta1 = 2;
    private static final int iBeta2 = 10;

    static Model        model;
    static NumericTable trainData;
    static NumericTable expectedResponses;
    static NumericTable predictedResponses;
    static NumericTable predictedReducedModelResponses;
    static ResultCollection qualityMetricSetResult;
    static double[] savedBetas;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        /* Initialize FileDataSource to retrieve the input data from a .csv file */
        FileDataSource trainDataSource = new FileDataSource(context, trainDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for training data and labels */
        trainData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
        expectedResponses = new HomogenNumericTable(context, Float.class, nDependentVariables, 0,
                                                    NumericTable.AllocationFlag.DoNotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(trainData);
        mergedData.addNumericTable(expectedResponses);

        /* Retrieve the data from an input file */
        trainDataSource.loadDataBlock(mergedData);

        for(int i = 0; i < 2; ++i)
        {
            if(i == 0)
            {
                /* Create an algorithm object to train the multiple linear regression model with normal equation method */
                System.out.println("Train model with normal equation algorithm.");
                /* Create an algorithm object to train the multiple linear regression model with the normal equations method */
                TrainingBatch linearRegressionTrain = new TrainingBatch(context, Float.class, TrainingMethod.normEqDense);
                        linearRegressionTrain.input.set(TrainingInputId.data, trainData);
        linearRegressionTrain.input.set(TrainingInputId.dependentVariable, expectedResponses);

        /* Build the multiple linear regression model */
        TrainingResult trainingResult = linearRegressionTrain.compute();

        model = trainingResult.get(TrainingResultId.model);

            }
            else
            {
                /* Create an algorithm object to train the multiple linear regression model with QR method */
                System.out.println("Train model with QR algorithm.");
                /* Create an algorithm object to train the multiple linear regression model with the normal equations method */
                TrainingBatch linearRegressionTrain = new TrainingBatch(context, Float.class, TrainingMethod.qrDense);
                        linearRegressionTrain.input.set(TrainingInputId.data, trainData);
        linearRegressionTrain.input.set(TrainingInputId.dependentVariable, expectedResponses);

        /* Build the multiple linear regression model */
        TrainingResult trainingResult = linearRegressionTrain.compute();

        model = trainingResult.get(TrainingResultId.model);

            }
            testModelQuality();
            printResults();
        }


        context.dispose();
    }

    private static NumericTable predictResults() {

        /* Create algorithm objects to predict values of multiple linear regression with the default method */
        PredictionBatch linearRegressionPredict = new PredictionBatch(context, Float.class, PredictionMethod.defaultDense);

        linearRegressionPredict.input.set(PredictionInputId.data, trainData);
        linearRegressionPredict.input.set(PredictionInputId.model, model);

        /* Compute prediction results */
        PredictionResult predictionResult = linearRegressionPredict.compute();

        return predictionResult.get(PredictionResultId.prediction);
    }

    private static void reduceModel() {
        final int nBeta = (int)model.getNumberOfBetas();
        savedBetas = new double[nBeta * nDependentVariables];

        /* Read a block of rows */
        DoubleBuffer betas = DoubleBuffer.allocate(nBeta * nDependentVariables);
        betas = model.getBeta().getBlockOfRows(0, nDependentVariables, betas);
        savedBetas[iBeta1] = betas.get(iBeta1);
        savedBetas[iBeta2] = betas.get(iBeta2);
        savedBetas[iBeta1 + nBeta] = betas.get(iBeta1 + nBeta);
        savedBetas[iBeta2 + nBeta] = betas.get(iBeta2 + nBeta);
        betas.put(iBeta1, 0);
        betas.put(iBeta2, 0);
        betas.put(iBeta1 + nBeta, 0);
        betas.put(iBeta2 + nBeta, 0);
        model.getBeta().releaseBlockOfRows(0, nDependentVariables, betas);
    }

    private static void restoreModel() {
        final int nBeta = (int)model.getNumberOfBetas();

        /* Read a block of rows */
        DoubleBuffer betas = DoubleBuffer.allocate(nBeta * nDependentVariables);
        betas = model.getBeta().getBlockOfRows(0, nDependentVariables, betas);
        betas.put(iBeta1, savedBetas[iBeta1]);
        betas.put(iBeta2, savedBetas[iBeta2]);
        betas.put(iBeta1 + nBeta, savedBetas[iBeta1 + nBeta]);
        betas.put(iBeta2 + nBeta, savedBetas[iBeta2 + nBeta]);
        model.getBeta().releaseBlockOfRows(0, nDependentVariables, betas);
    }

    private static void testModelQuality() {

        /* Compute prediction results */
        predictedResponses = predictResults();

        /* Predict results with the reduced model */
        reduceModel();
        predictedReducedModelResponses = predictResults();
        restoreModel();

        /* Create a quality metric set object to compute quality metrics of the linear regression algorithm */
        final long nBeta = model.getNumberOfBetas();
        final long nBetaReducedModel = nBeta - 2;

        QualityMetricSetBatch qms = new QualityMetricSetBatch(context, nBeta, nBetaReducedModel);

        SingleBetaInput singleBetaInput = (SingleBetaInput)qms.getInputDataCollection().getInput(QualityMetricId.singleBeta);
        singleBetaInput.set(SingleBetaModelInputId.model, model);
        singleBetaInput.set(SingleBetaDataInputId.expectedResponses, expectedResponses);
        singleBetaInput.set(SingleBetaDataInputId.predictedResponses, predictedResponses);

        GroupOfBetasInput groupOfBetasInput = (GroupOfBetasInput)qms.getInputDataCollection().getInput(QualityMetricId.groupOfBetas);
        groupOfBetasInput.set(GroupOfBetasInputId.expectedResponses, expectedResponses);
        groupOfBetasInput.set(GroupOfBetasInputId.predictedResponses, predictedResponses);
        groupOfBetasInput.set(GroupOfBetasInputId.predictedReducedModelResponses, predictedReducedModelResponses);

        /* Compute quality metrics */
        qualityMetricSetResult = qms.compute();
    }

    private static void printResults() {
        NumericTable beta = model.getBeta();
        Service.printNumericTable("Linear Regression coefficients:", beta);
        Service.printNumericTable("Expected responses (first 20 rows):", expectedResponses, 20);
        Service.printNumericTable("Predicted responses (first 20 rows):", predictedResponses, 20);
        Service.printNumericTable("Responses predicted with reduced model (first 20 rows):", predictedReducedModelResponses, 20);

        /* Print the quality metrics for a single beta */
        System.out.println("Quality metrics for a single beta");
        SingleBetaResult singleBetaResult = (SingleBetaResult)qualityMetricSetResult.getResult(QualityMetricId.singleBeta);

        Service.printNumericTable("Root means square errors for each response (dependent variable):", singleBetaResult.get(SingleBetaResultId.rms), 20);
        Service.printNumericTable("Variance for each response (dependent variable):", singleBetaResult.get(SingleBetaResultId.variance), 20);
        Service.printNumericTable("Z-score statistics:", singleBetaResult.get(SingleBetaResultId.zScore), 20);
        Service.printNumericTable("Confidence intervals for each beta coefficient:", singleBetaResult.get(SingleBetaResultId.confidenceIntervals), 20);
        Service.printNumericTable("Inverse(Xt * X) matrix:", singleBetaResult.get(SingleBetaResultId.inverseOfXtX), 20);

        DataCollection coll = singleBetaResult.get(SingleBetaResultDataCollectionId.betaCovariances);
        for (int i = 0; i < coll.size(); i++) {
            NumericTable tbl = (NumericTable)coll.get(i);
            Service.printNumericTable("Variance-covariance matrix for betas of " + i + "-th response\n", tbl, 20);
        }

        /* Print quality metrics for a group of betas */
        System.out.println("Quality metrics for a group of betas");
        GroupOfBetasResult groupOfBetasResult = (GroupOfBetasResult)qualityMetricSetResult.getResult(QualityMetricId.groupOfBetas);
        Service.printNumericTable("Means of expected responses for each dependent variable:", groupOfBetasResult.get(GroupOfBetasResultId.expectedMeans), 20);
        Service.printNumericTable("Variance of expected responses for each dependent variable:", groupOfBetasResult.get(GroupOfBetasResultId.expectedVariance), 20);
        Service.printNumericTable("Regression sum of squares of expected responses:", groupOfBetasResult.get(GroupOfBetasResultId.regSS), 20);
        Service.printNumericTable("Sum of squares of residuals for each dependent variable:", groupOfBetasResult.get(GroupOfBetasResultId.resSS), 20);
        Service.printNumericTable("Total sum of squares for each dependent variable:", groupOfBetasResult.get(GroupOfBetasResultId.tSS), 20);
        Service.printNumericTable("Determination coefficient for each dependent variable:", groupOfBetasResult.get(GroupOfBetasResultId.determinationCoeff), 20);
        Service.printNumericTable("F-statistics for each dependent variable:", groupOfBetasResult.get(GroupOfBetasResultId.fStatistics), 20);
    }
}
