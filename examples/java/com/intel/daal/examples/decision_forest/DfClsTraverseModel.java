/* file: DfClsTraverseModel.java */
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
 //     Java example of decision forest classification model traversal
 //
 //     The program trains the decision forest classification model on a training
//      datasetFileName and prints the trained model by its depth-first traversing.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-DfClsTraverseModel">
 * @example DfClsTraverseModel.java
 */

package com.intel.daal.examples.decision_forest;

import com.intel.daal.algorithms.tree_utils.classification.TreeNodeVisitor;
import com.intel.daal.algorithms.tree_utils.classification.LeafNodeDescriptor;
import com.intel.daal.algorithms.tree_utils.SplitNodeDescriptor;
import com.intel.daal.algorithms.classifier.training.InputId;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.decision_forest.classification.Model;
import com.intel.daal.algorithms.decision_forest.classification.training.*;
import com.intel.daal.algorithms.decision_forest.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.*;

class DfClsPrintNodeVisitor extends TreeNodeVisitor {
    @Override
    public boolean onLeafNode(LeafNodeDescriptor desc) {
        if(desc.level != 0)
            printTab(desc.level);
        System.out.println("Level " + desc.level + ", leaf node. Response value = " + desc.label +
            ", Impurity = " + desc.impurity + ", nNodeSampleCount = " + desc.nNodeSampleCount);
        return true;
    }

    public boolean onSplitNode(SplitNodeDescriptor desc){
        if(desc.level != 0)
            printTab(desc.level);
        System.out.println("Level " + desc.level + ", split node. Feature index = " + desc.featureIndex + ", feature value = " + desc.featureValue +
            ", Impurity = " + desc.impurity + ", nNodeSampleCount = " + desc.nNodeSampleCount);
        return true;
    }

    private void printTab(long level) {
        String s = "";
        for (long i = 0; i < level; i++) {
            s += "  ";
        }
        System.out.print(s);
    }
}

class DfClsTraverseModel {
    /* Input data set parameters */
    private static final String trainDataset = "../data/batch/df_classification_train.csv";

    private static final int nFeatures     = 3;
    private static final int nClasses      = 5;

    /* Decision forest classification algorithm parameters */
    private static final int nTrees = 2;
    private static final int minObservationsInLeafNode = 8;
    private static final int maxTreeDepth = 15;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        TrainingResult trainingResult = trainModel();
        printModel(trainingResult);
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

        /* Create algorithm objects to train the decision forest classification model */
        TrainingBatch algorithm = new TrainingBatch(context, Float.class, TrainingMethod.defaultDense, nClasses);
        algorithm.parameter.setNTrees(nTrees);
        algorithm.parameter.setFeaturesPerNode(nFeatures);
        algorithm.parameter.setMinObservationsInLeafNode(minObservationsInLeafNode);
        algorithm.parameter.setMaxTreeDepth(maxTreeDepth);

        /* Pass a training data set and dependent values to the algorithm */
        algorithm.input.set(InputId.data, trainData);
        algorithm.input.set(InputId.labels, trainGroundTruth);

        /* Train the decision forest classification model */
        return algorithm.compute();
    }

    private static void printModel(TrainingResult trainingResult) {
        Model m = trainingResult.get(TrainingResultId.model);
        long nTrees = m.getNumberOfTrees();
        System.out.println("Number of trees: " + nTrees);
        DfClsPrintNodeVisitor visitor = new DfClsPrintNodeVisitor();
        for (long i = 0; i < nTrees; i++) {
            System.out.println("Tree #" + i);
            m.traverseDFS(i, visitor);
        }
    }
}
