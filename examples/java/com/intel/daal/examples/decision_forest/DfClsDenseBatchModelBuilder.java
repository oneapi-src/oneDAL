/* file: DfClsDenseBatchModelBuilder.java */
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
 //     Java example of decision forest classification model building.
 //
 //     The program builds the decision forest classification model
 //     via Model Builder and computes classification for the test data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-DfClsDenseBatchModelBuilder">
 * @example DfClsDenseBatchModelBuilder.java
 */

package com.intel.daal.examples.decision_forest;

import com.intel.daal.algorithms.classifier.prediction.ModelInputId;
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId;
import com.intel.daal.algorithms.classifier.prediction.PredictionResult;
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId;
import com.intel.daal.algorithms.decision_forest.classification.Model;
import com.intel.daal.algorithms.decision_forest.classification.ModelBuilder;
import com.intel.daal.algorithms.decision_forest.classification.prediction.*;
import com.intel.daal.algorithms.decision_forest.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.*;

class DfClsDenseBatchModelBuilder {
    /* Input data set parameters */
    private static final String testDatasetFileName = "../data/batch/df_classification_model_builder_test.csv";

    private static final long categoricalFeaturesIndices [] =  { 2 };

    /* Number of features in training and testing data sets */
    private static final int nFeatures     = 3;
    /* Number of classes */
    private static final int nClasses      = 5;
    /* Number of tree in decision forest classification model */
    private static final int nTrees = 3;
    private static final int minObservationsInLeafNode = 8;

    private static NumericTable testGroundTruth;
    static PredictionResult predictionResult;
    static Model model;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        buildModel();
        testModel();
        printResults(predictionResult);

        context.dispose();
    }

    public static void buildModel() {
        final long nNodes = 3;

        ModelBuilder modelBuilder = new ModelBuilder(context, nClasses, nTrees);

        long tree1 = modelBuilder.createTree(nNodes);
        long root1   = modelBuilder.addSplitNode(tree1, ModelBuilder.noParent, 0, 0, 0.174108);
        long child11 = modelBuilder.addLeafNode(tree1, root1, 0, 0);
        long child12 = modelBuilder.addLeafNode(tree1, root1, 1, 4);

        long tree2 = modelBuilder.createTree(nNodes);
        long root2 = modelBuilder.addSplitNode(tree2, modelBuilder.noParent, 0, 1, 0.571184);
        long child22 = modelBuilder.addLeafNode(tree2, root2, 1, 4);
        long child21 = modelBuilder.addLeafNode(tree2, root2, 0, 2);
        long tree3 = modelBuilder.createTree(nNodes);
        long root3 = modelBuilder.addSplitNode(tree3, modelBuilder.noParent, 0, 0, 0.303995);
        long child32 = modelBuilder.addLeafNode(tree3, root3, 1, 4);
        long child31 = modelBuilder.addLeafNode(tree3, root3, 0, 2);

        model = modelBuilder.getModel();
    }

    private static void testModel() {

        /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
        FileDataSource testDataSource = new FileDataSource(context, testDatasetFileName,
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
        testData.getDictionary().setFeature(Float.class, 2, DataFeatureUtils.FeatureType.DAAL_CATEGORICAL);

        /* Create algorithm objects for decision forest classification prediction with the fast method */
        PredictionBatch algorithm = new PredictionBatch(context, Float.class, PredictionMethod.defaultDense, nClasses);

        /* Pass a testing data set and the trained model to the algorithm */
        algorithm.input.set(NumericTableInputId.data, testData);
        algorithm.input.set(ModelInputId.model, model);

        /* Compute prediction results */
        predictionResult = algorithm.compute();
    }

    private static void printResults(PredictionResult predictionResult) {
        NumericTable predictionResults = predictionResult.get(PredictionResultId.prediction);
        Service.printNumericTable("Decision forest prediction results (first 10 rows):", predictionResults, 10);
        Service.printNumericTable("Ground truth (first 10 rows):", testGroundTruth, 10);
    }

}
