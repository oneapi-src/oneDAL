/* file: MnNaiveBayesDenseDistr.java */
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
 //     Java example of Naive Bayes classification in the distributed processing
 //     mode.
 //
 //     The program trains the Naive Bayes model on a supplied training data set
 //     in dense format and then performs classification of previously unseen
 //     data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-MULTINOMIALNAIVEBAYESDENSEDISTRIBUTED">
 * @example MnNaiveBayesDenseDistr.java
 */

package com.intel.daal.examples.naive_bayes;

import com.intel.daal.algorithms.classifier.prediction.ModelInputId;
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId;
import com.intel.daal.algorithms.classifier.prediction.PredictionResult;
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId;
import com.intel.daal.algorithms.classifier.training.InputId;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.multinomial_naive_bayes.Model;
import com.intel.daal.algorithms.multinomial_naive_bayes.prediction.*;
import com.intel.daal.algorithms.multinomial_naive_bayes.training.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class MnNaiveBayesDenseDistr {

    /* Input data set parameters */
    private static final String[] trainDatasetFileNames = { "../data/distributed/naivebayes_train_dense_1.csv",
            "../data/distributed/naivebayes_train_dense_2.csv", "../data/distributed/naivebayes_train_dense_3.csv",
            "../data/distributed/naivebayes_train_dense_4.csv" };

    private static final String testDatasetFileName = "../data/distributed/naivebayes_test_dense.csv";

    private static final int  nFeatures            = 20;
    private static final int  nBlocks              = 4;
    private static final long nClasses             = 20;

    /* Parameters for the Naive Bayes algorithm */
    private static TrainingResult   trainingResult;
    private static PredictionResult predictionResult;
    private static NumericTable     testGroundTruth;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        trainModel();

        testModel();

        printResults();
        context.dispose();
    }

    private static void trainModel() throws java.io.FileNotFoundException, java.io.IOException {
        TrainingPartialResult[] pres = new TrainingPartialResult[nBlocks];

        for (int node = 0; node < nBlocks; node++) {
            DaalContext localContext = new DaalContext();

            /* Initialize FileDataSource to retrieve the input data from a .csv file */
            FileDataSource trainDataSource = new FileDataSource(localContext, trainDatasetFileNames[node],
                    DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                    DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

            /* Create Numeric Tables for training data and labels */
            NumericTable trainData = new HomogenNumericTable(localContext, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
            NumericTable trainGroundTruth = new HomogenNumericTable(localContext, Float.class, 1, 0, NumericTable.AllocationFlag.DoNotAllocate);
            MergedNumericTable mergedData = new MergedNumericTable(localContext);
            mergedData.addNumericTable(trainData);
            mergedData.addNumericTable(trainGroundTruth);

            /* Retrieve the data from an input file */
            trainDataSource.loadDataBlock(mergedData);

            /* Create algorithm objects to train the Naive Bayes model */
            TrainingDistributedStep1Local algorithm = new TrainingDistributedStep1Local(localContext, Float.class,
                    TrainingMethod.defaultDense, nClasses);

            /* Set the input data */
            algorithm.input.set(InputId.data, trainData);
            algorithm.input.set(InputId.labels, trainGroundTruth);

            /* Build a partial Naive Bayes model */
            pres[node] = algorithm.compute();

            pres[node].changeContext(context);

            localContext.dispose();
        }

        /* Build the final Naive Bayes model on the master node*/
        TrainingDistributedStep2Master algorithm = new TrainingDistributedStep2Master(context, Float.class,
                TrainingMethod.defaultDense, nClasses);

        /* Set partial Naive Bayes models built on local nodes */
        for (int node = 0; node < nBlocks; node++) {
            algorithm.input.add(TrainingDistributedInputId.partialModels, pres[node]);
        }

        /* Build the final Naive Bayes model */
        algorithm.compute();

        trainingResult = algorithm.finalizeCompute();
    }

    private static void testModel() throws java.io.FileNotFoundException, java.io.IOException {
        FileDataSource testDataSource = new FileDataSource(context, testDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for testing data and labels */
        NumericTable testData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
        testGroundTruth = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.DoNotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(testData);
        mergedData.addNumericTable(testGroundTruth);

        /* Retrieve the data from an input file */
        testDataSource.loadDataBlock(mergedData);

        /* Create algorithm objects to predict Naive Bayes values with the defaultDense method */
        PredictionBatch algorithm = new PredictionBatch(context, Float.class, PredictionMethod.defaultDense, nClasses);

        /* Pass a testing data set and the trained model to the algorithm */
        algorithm.input.set(NumericTableInputId.data, testData);
        Model model = trainingResult.get(TrainingResultId.model);
        algorithm.input.set(ModelInputId.model, model);

        /* Compute the prediction results */
        predictionResult = algorithm.compute();
    }

    private static void printResults() throws java.io.FileNotFoundException, java.io.IOException {
        NumericTable expected = testGroundTruth;
        NumericTable prediction = predictionResult.get(PredictionResultId.prediction);
        Service.printClassificationResult(expected, prediction, "Ground truth", "Classification results",
                "NaiveBayes classification results (first 20 observations):", 20);
        System.out.println("");
    }
}
