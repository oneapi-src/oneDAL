/* file: DtClsTraverseModel.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
 //     Java example of decision tree classification model traversal
 //
 //     The program trains the decision tree classification model on a training
//      datasetFileName and prints the trained model by its depth-first traversing.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-DtClsTraverseModel">
 * @example DtClsTraverseModel.java
 */

package com.intel.daal.examples.decision_tree;

import com.intel.daal.algorithms.tree_utils.classification.TreeNodeVisitor;
import com.intel.daal.algorithms.tree_utils.classification.LeafNodeDescriptor;
import com.intel.daal.algorithms.tree_utils.SplitNodeDescriptor;
import com.intel.daal.algorithms.classifier.prediction.PredictionResult;
import com.intel.daal.algorithms.classifier.prediction.ModelInputId;
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId;
import com.intel.daal.algorithms.classifier.training.InputId;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId;
import com.intel.daal.algorithms.decision_tree.classification.Model;
import com.intel.daal.algorithms.decision_tree.classification.prediction.*;
import com.intel.daal.algorithms.decision_tree.classification.training.*;
import com.intel.daal.algorithms.decision_tree.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.*;

class DtClsPrintNodeVisitor extends TreeNodeVisitor {
    @Override
    public boolean onLeafNode(LeafNodeDescriptor desc) {
        if(desc.level != 0)
            printTab(desc.level);
        System.out.println("Level " + desc.level + ", leaf node. Response value = " + desc.label +
            ", Impurity = " + desc.impurity + ", Number of samples = " + desc.nNodeSampleCount);
        return true;
    }

    public boolean onSplitNode(SplitNodeDescriptor desc){
        if(desc.level != 0)
            printTab(desc.level);
        System.out.println("Level " + desc.level + ", split node. Feature index = " + desc.featureIndex + ", feature value = " + desc.featureValue +
            ", Impurity = " + desc.impurity + ", Number of samples = " + desc.nNodeSampleCount);
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

class DtClsTraverseModel {
    /* Input data set parameters */
    private static final String trainDataset = "../data/batch/decision_tree_train.csv";
    private static final String pruneDataset = "../data/batch/decision_tree_prune.csv";

    private static final int nFeatures     = 5; /* Number of features in training and testing data sets */
    private static final int nClasses      = 5; /* Number of classes */

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

        /* Retrieve the pruning data from the input data sets */
        FileDataSource pruneDataSource = new FileDataSource(context, pruneDataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for pruning data and labels */
        NumericTable pruneData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.NotAllocate);
        NumericTable pruneGroundTruth = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.NotAllocate);
        MergedNumericTable pruneMergedData = new MergedNumericTable(context);
        pruneMergedData.addNumericTable(pruneData);
        pruneMergedData.addNumericTable(pruneGroundTruth);

        /* Retrieve the pruning data from an input file */
        pruneDataSource.loadDataBlock(pruneMergedData);

        /* Create algorithm objects to train the decision tree classification model */
        TrainingBatch algorithm = new TrainingBatch(context, Float.class, TrainingMethod.defaultDense, nClasses);

        /* Pass the training data set with labels, and pruning dataset with labels to the algorithm */
        algorithm.input.set(InputId.data, trainData);
        algorithm.input.set(InputId.labels, trainGroundTruth);
        algorithm.input.set(TrainingInputId.dataForPruning, pruneData);
        algorithm.input.set(TrainingInputId.labelsForPruning, pruneGroundTruth);

        /* Train the decision forest classification model */
        return algorithm.compute();
    }

    private static void printModel(TrainingResult trainingResult) {
        Model m = trainingResult.get(TrainingResultId.model);
        DtClsPrintNodeVisitor visitor = new DtClsPrintNodeVisitor();
        m.traverseDFS(visitor);
    }
}
