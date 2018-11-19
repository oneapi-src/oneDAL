/* file: GbtClsDenseBatch.java */
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
 //     Java example of gradient boosted trees classification.
 //
 //     The program trains the gradient boosted trees classification model on a supplied
 //     training data set and then performs classification of previously unseen data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-GbtClsDenseBatch">
 * @example GbtClsDenseBatch.java
 */

package com.intel.daal.examples.gbt;

import com.intel.daal.algorithms.classifier.prediction.ModelInputId;
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId;
import com.intel.daal.algorithms.classifier.training.InputId;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.classifier.prediction.PredictionResult;
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId;
import com.intel.daal.algorithms.gbt.classification.Model;
import com.intel.daal.algorithms.gbt.classification.prediction.*;
import com.intel.daal.algorithms.gbt.classification.training.*;
import com.intel.daal.algorithms.gbt.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.*;

class GbtClsDenseBatch {
    /* Input data set parameters */
    private static final String trainDataset = "../data/batch/df_classification_train.csv";

    private static final String testDataset  = "../data/batch/df_classification_test.csv";

    private static final int nFeatures     = 3;
    private static final int nClasses      = 5;

    /* Gradient boosted trees classification algorithm parameters */
    private static final int maxIterations = 40;
    private static final int minObservationsInLeafNode = 8;

    private static NumericTable testGroundTruth;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        TrainingResult trainingResult = trainModel();

        PredictionResult predictionResult = testModel(trainingResult);

        printResults(predictionResult);

        context.dispose();
    }

    private static TrainingResult trainModel() {
        /* Retrieve the data from the input data sets */
        FileDataSource trainDataSource = new FileDataSource(context, trainDataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for training data and labels */
        NumericTable trainData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.NotAllocate);
        NumericTable trainGroundTruth = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.NotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(trainData);
        mergedData.addNumericTable(trainGroundTruth);

        /* Retrieve the data from an input file */
        trainDataSource.loadDataBlock(mergedData);

        /* Set feature as categorical */
        trainData.getDictionary().setFeature(Float.class,2,DataFeatureUtils.FeatureType.DAAL_CATEGORICAL);

        /* Create algorithm objects to train the gradient boosted trees classification model */
        TrainingBatch algorithm = new TrainingBatch(context, Float.class, TrainingMethod.defaultDense, nClasses);
        algorithm.parameter.setMaxIterations(maxIterations);
        algorithm.parameter.setFeaturesPerNode(nFeatures);
        algorithm.parameter.setMinObservationsInLeafNode(minObservationsInLeafNode);

        /* Pass a training data set and dependent values to the algorithm */
        algorithm.input.set(InputId.data, trainData);
        algorithm.input.set(InputId.labels, trainGroundTruth);

        /* Train the gradient boosted trees classification model */
        TrainingResult trainingResult = algorithm.compute();
        return trainingResult;
    }

    private static PredictionResult testModel(TrainingResult trainingResult) {
        FileDataSource testDataSource = new FileDataSource(context, testDataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for testing data and labels */
        NumericTable testData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.NotAllocate);
        testGroundTruth = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.NotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(testData);
        mergedData.addNumericTable(testGroundTruth);

        /* Retrieve the data from an input file */
        testDataSource.loadDataBlock(mergedData);
        /* Set feature as categorical */
        testData.getDictionary().setFeature(Float.class,2,DataFeatureUtils.FeatureType.DAAL_CATEGORICAL);

        /* Create algorithm objects for gradient boosted trees classification prediction with the fast method */
        PredictionBatch algorithm = new PredictionBatch(context, Float.class, PredictionMethod.defaultDense, nClasses);

        /* Pass a testing data set and the trained model to the algorithm */
        Model model = trainingResult.get(TrainingResultId.model);
        algorithm.input.set(NumericTableInputId.data, testData);
        algorithm.input.set(ModelInputId.model, model);

        /* Compute prediction results */
        return algorithm.compute();
    }

    private static void printResults(PredictionResult predictionResult) {
        NumericTable predictionResults = predictionResult.get(PredictionResultId.prediction);
        Service.printNumericTable("Gragient boosted trees prediction results (first 10 rows):", predictionResults, 10);
        Service.printNumericTable("Ground truth (first 10 rows):", testGroundTruth, 10);
    }

}
