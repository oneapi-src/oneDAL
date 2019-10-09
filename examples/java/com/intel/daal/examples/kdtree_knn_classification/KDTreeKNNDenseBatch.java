/* file: KDTreeKNNDenseBatch.java */
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
 //     Java example of k nearest neighbors algorithm in the batch processing mode.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-KDTREEKNNDENSEBATCH">
 * @example KDTreeKNNDenseBatch.java
 */

package com.intel.daal.examples.kdtree_knn_classification;

import com.intel.daal.algorithms.kdtree_knn_classification.Model;
import com.intel.daal.algorithms.kdtree_knn_classification.prediction.*;
import com.intel.daal.algorithms.kdtree_knn_classification.training.*;
import com.intel.daal.algorithms.classifier.training.InputId;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.classifier.prediction.ModelInputId;
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId;
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId;
import com.intel.daal.algorithms.classifier.prediction.PredictionResult;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class KDTreeKNNDenseBatch {
    /* Input data set parameters */
    private static final String trainDatasetFileName = "../data/batch/k_nearest_neighbors_train.csv";

    private static final String testDatasetFileName  = "../data/batch/k_nearest_neighbors_test.csv";

    private static final int nFeatures           = 5;
    private static final long nClasses           = 5;

    static Model        model;
    static NumericTable results;
    static NumericTable testGroundTruth;

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
        NumericTable trainGroundTruth = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.DoNotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(trainData);
        mergedData.addNumericTable(trainGroundTruth);

        /* Retrieve the data from an input file */
        trainDataSource.loadDataBlock(mergedData);

        /* Create an algorithm object to train the k nearest neighbors model with the default dense method */
        TrainingBatch kNearestNeighborsTrain = new TrainingBatch(context, Float.class, TrainingMethod.defaultDense);

        kNearestNeighborsTrain.input.set(InputId.data, trainData);
        kNearestNeighborsTrain.input.set(InputId.labels, trainGroundTruth);
        kNearestNeighborsTrain.parameter.setNClasses(nClasses);

        /* Build the k nearest neighbors model */
        TrainingResult trainingResult = kNearestNeighborsTrain.compute();

        model = trainingResult.get(TrainingResultId.model);
    }

    private static void testModel() {
        /* Initialize FileDataSource to retrieve the input data from a .csv file */
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

        /* Create algorithm objects to predict values of k nearest neighbors with the default method */
        PredictionBatch kNearestNeighborsPredict = new PredictionBatch(context, Float.class,
                PredictionMethod.defaultDense);

        kNearestNeighborsPredict.input.set(NumericTableInputId.data, testData);
        kNearestNeighborsPredict.input.set(ModelInputId.model, model);

        /* Compute prediction results */
        PredictionResult predictionResult = kNearestNeighborsPredict.compute();

        results = predictionResult.get(PredictionResultId.prediction);
    }

    private static void printResults() {
        NumericTable expected = testGroundTruth;
        Service.printClassificationResult(expected,results,"Ground truth","Classification results","KD-tree based kNN classification results (first 20 observations):",20);
        System.out.println("");
    }
}
