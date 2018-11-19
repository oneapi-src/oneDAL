/* file: MnNaiveBayesCSROnline.java */
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
 //     Java example of Naive Bayes classification in the online processing mode.
 //
 //     The program trains the Naive Bayes model on a supplied training data set
 //     in compressed sparse rows (CSR) format and then performs classification
 //     of previously unseen data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-MULTINOMIALNAIVEBAYESCSRONLINE">
 * @example MnNaiveBayesCSROnline.java
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
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class MnNaiveBayesCSROnline {

    /* Input data set parameters */
    private static final String[] trainGroundTruthFileNames = { "../data/online/naivebayes_train_labels_1.csv",
            "../data/online/naivebayes_train_labels_2.csv", "../data/online/naivebayes_train_labels_3.csv",
            "../data/online/naivebayes_train_labels_4.csv" };
    private static final String[] trainDatasetFileNames     = { "../data/online/naivebayes_train_csr_1.csv",
            "../data/online/naivebayes_train_csr_2.csv", "../data/online/naivebayes_train_csr_3.csv",
            "../data/online/naivebayes_train_csr_4.csv" };

    private static final String testDatasetFileName     = "../data/online/naivebayes_test_csr.csv";
    private static final String testGroundTruthFileName = "../data/online/naivebayes_test_labels.csv";

    private static final int  nTrainObservations = 8000;
    private static final int  nTestObservations  = 2000;
    private static final long nClasses           = 20;
    private static final int  nBlocks            = 4;

    /* Parameters for the Naive Bayes algorithm */
    private static TrainingResult   trainingResult;
    private static PredictionResult predictionResult;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        trainModel();

        testModel();

        printResults();
        context.dispose();
    }

    private static void trainModel() throws java.io.FileNotFoundException, java.io.IOException {
        /* Create algorithm objects to train the Naive Bayes model */
        TrainingOnline algorithm = new TrainingOnline(context, Float.class, TrainingMethod.fastCSR, nClasses);
        for (int node = 0; node < nBlocks; node++) {
        /* Initialize FileDataSource to retrieve the input data from a .csv file */
            FileDataSource trainGroundTruthSource = new FileDataSource(context, trainGroundTruthFileNames[node],
                    DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                    DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

            /* Create Numeric Tables for training data and labels */
            CSRNumericTable trainData = Service.createSparseTable(context, trainDatasetFileNames[node]);
            NumericTable labels = trainGroundTruthSource.getNumericTable();

            /* Retrieve the data from input file */
            trainGroundTruthSource.loadDataBlock(nTrainObservations);

            /* Pass a training data set and dependent values to the algorithm */
            algorithm.input.set(InputId.data,   trainData);
            algorithm.input.set(InputId.labels, labels);

            /* Train the Naive Bayes model */
            algorithm.compute();
        }

        /* Retrieve the algorithm results */
        trainingResult = algorithm.finalizeCompute();
    }

    private static void testModel() throws java.io.FileNotFoundException, java.io.IOException {
        /* Create algorithm objects to predict Naive Bayes values with the fastCSR method */
        PredictionBatch algorithm = new PredictionBatch(context, Float.class, PredictionMethod.fastCSR, nClasses);

        /* Create Numeric Table for test data */
        CSRNumericTable testData = Service.createSparseTable(context, testDatasetFileName);

        /* Pass a testing data set and the trained model to the algorithm */
        algorithm.input.set(NumericTableInputId.data, testData);
        Model model = trainingResult.get(TrainingResultId.model);
        algorithm.input.set(ModelInputId.model, model);

        /* Compute the prediction results */
        predictionResult = algorithm.compute();
    }

    private static void printResults() throws java.io.FileNotFoundException, java.io.IOException {

        FileDataSource testGroundTruth = new FileDataSource(context, testGroundTruthFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        testGroundTruth.loadDataBlock(nTestObservations);

        NumericTable expected = testGroundTruth.getNumericTable();
        NumericTable prediction = predictionResult.get(PredictionResultId.prediction);
        Service.printClassificationResult(expected, prediction, "Ground truth", "Classification results",
                "NaiveBayes classification results (first 20 observations):", 20);
        System.out.println("");
    }
}
