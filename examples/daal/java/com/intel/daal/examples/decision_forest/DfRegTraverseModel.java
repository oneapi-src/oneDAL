/* file: DfRegTraverseModel.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
 //     Java example of decision forest regression model traversal
 //
 //     The program trains the decision forest regression model on a training
 //     datasetFileName and prints the trained model by its depth-first traversing.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-DfRegTraverseModel">
 * @example DfRegTraverseModel.java
 */

package com.intel.daal.examples.decision_forest;

import com.intel.daal.algorithms.tree_utils.regression.TreeNodeVisitor;
import com.intel.daal.algorithms.tree_utils.regression.LeafNodeDescriptor;
import com.intel.daal.algorithms.tree_utils.SplitNodeDescriptor;
import com.intel.daal.algorithms.decision_forest.regression.*;
import com.intel.daal.algorithms.decision_forest.regression.training.*;
import com.intel.daal.algorithms.decision_forest.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.*;

class DfRegPrintNodeVisitor extends TreeNodeVisitor {
    @Override
    public boolean onLeafNode(LeafNodeDescriptor desc) {
        if(desc.level != 0)
            printTab(desc.level);
        System.out.println("Level " + desc.level + ", leaf node. Response value = " + desc.response +
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
        StringBuffer s = new StringBuffer();
        for (long i = 0; i < level; i++) {
            s.append("  ");
        }
        System.out.print(s.toString());
    }
}

class DfRegTraverseModel {
    /* Input data set parameters */
    private static final String trainDataset = "../data/batch/df_regression_train.csv";

    private static final int nFeatures     = 13;

    /* Decision forest regression algorithm parameters */
    private static final int nTrees = 2;

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
        trainData.getDictionary().setFeature(Float.class,3,DataFeatureUtils.FeatureType.DAAL_CATEGORICAL);

        /* Create algorithm objects to train the decision forest regression model */
        TrainingBatch algorithm = new TrainingBatch(context, Float.class, TrainingMethod.defaultDense);
        algorithm.parameter.setNTrees(nTrees);

        /* Pass a training data set and dependent values to the algorithm */
        algorithm.input.set(InputId.data, trainData);
        algorithm.input.set(InputId.dependentVariable, trainGroundTruth);

        /* Train the decision forest regression model */
        return algorithm.compute();
    }

    private static void printModel(TrainingResult trainingResult) {
        Model m = trainingResult.get(TrainingResultId.model);
        long nTrees = m.getNumberOfTrees();
        System.out.println("Number of trees: " + nTrees);
        DfRegPrintNodeVisitor visitor = new DfRegPrintNodeVisitor();
        for (long i = 0; i < nTrees; i++) {
            System.out.println("Tree #" + i);
            m.traverseDFS(i, visitor);
        }
    }
}
