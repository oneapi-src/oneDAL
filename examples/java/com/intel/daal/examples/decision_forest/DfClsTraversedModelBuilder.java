/* file: DfClsTraversedModelBuilder.java */
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
 //     Java example of decision forest classification model building from
 //     traversed decision forest algorithm.
 //
 //     The program trains the decision forest classification model, gets
 //     pre-computed values from nodes of each tree using traversal and build
 //     model of the decision forest classification via Model Builder and
 //     computes classification for the test data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-DFCLSTRAVERSEDMODELBUILDER">
 * @example DfClsTraversedModelBuilder.java
 */

package com.intel.daal.examples.decision_forest;

import com.intel.daal.algorithms.tree_utils.classification.TreeNodeVisitor;
import com.intel.daal.algorithms.tree_utils.classification.LeafNodeDescriptor;
import com.intel.daal.algorithms.tree_utils.SplitNodeDescriptor;
import com.intel.daal.algorithms.classifier.training.InputId;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.classifier.prediction.ModelInputId;
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId;
import com.intel.daal.algorithms.classifier.prediction.PredictionResult;
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId;
import com.intel.daal.algorithms.decision_forest.classification.Model;
import com.intel.daal.algorithms.decision_forest.classification.ModelBuilder;
import com.intel.daal.algorithms.decision_forest.classification.prediction.*;
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
import java.util.LinkedList;
import java.util.Queue;
import java.util.HashMap;
import java.util.Map;


/* Node structure for representing nodes in trees after traversing DAAL model */
class Node {
    public Node left;
    public Node right;
    public long classLabel;
    public long featureIndex;
    public double featureValue;

    Node(long cl, long fi, double fv) {
        left = null;
        right = null;
        classLabel = cl;
        featureIndex = fi;
        featureValue = fv;
    }

    Node() {
        left = null;
        right = null;
        classLabel = 0;
        featureIndex = 0;
        featureValue = 0;
    }
}

/* Tree structure for representing tree after traversing DAAL model */
class Tree {
    public Node root;
    public long nNodes;
}

/* Example of structure to remember relationship between nodes */
class ParentPlace {
    public long parentId;
    public long place;

    ParentPlace(long _parent, long _place) {
        parentId = _parent;
        place = _place;
    }

    ParentPlace() {
        parentId = 0;
        place = 0;
    }
}

/* Visitor class implementing TreeNodeVisitor interface, prints out tree nodes of the model when it is called back by model traversal method */
class BFSNodeVisitor extends TreeNodeVisitor {

    public Tree [] roots;
    int treeId;
    Queue<Node> parentNodes;

    BFSNodeVisitor(int nTrees) {
        roots = new Tree[nTrees];
        for(int i = 0; i < nTrees; i++) {
            roots[i] = new Tree();
            roots[i].root = new Node();
        }
        treeId = 0;
        parentNodes = new LinkedList<Node>();
    }

    @Override
    public boolean onLeafNode(LeafNodeDescriptor desc) {
        if(desc.level == 0) {
            Node root = roots[treeId].root;
            roots[treeId].nNodes = 1;
            root.left = null;
            root.right = null;
            root.classLabel = desc.label;
            root.featureIndex = 0;
            root.featureValue = 0;
            treeId++;
        }
        else {
            roots[treeId - 1].nNodes++;
            Node node = new Node(desc.label, 0, 0);

            Node parent = parentNodes.peek();
            if(parent.left == null) {
                parent.left = node;
            }
            else {
                parent.right = node;
                parentNodes.poll();
            }
        }
        return true;
    }

    @Override
    public boolean onSplitNode(SplitNodeDescriptor desc) {
        if(desc.level == 0) {
            Node root = roots[treeId].root;
            roots[treeId].nNodes = 1;
            root.left = null;
            root.right = null;
            root.classLabel = 0;
            root.featureIndex = desc.featureIndex;
            root.featureValue = desc.featureValue;
            parentNodes.add(root);
            treeId++;
        }
        else {
            roots[treeId - 1].nNodes++;
            Node node = new Node(0,desc.featureIndex,desc.featureValue);

            Node parent = parentNodes.peek();
            if(parent.left == null) {
                parent.left = node;
            }
            else {
                parent.right = node;
                parentNodes.poll();
            }
            parentNodes.add(node);
        }
        return true;
    }

}

class DfClsTraversedModelBuilder {
    /* Input data set parameters */
    private static final String trainDatasetFileName = "../data/batch/df_classification_train.csv";
    private static final String testDatasetFileName = "../data/batch/df_classification_test.csv";
    private static final long categoricalFeaturesIndices [] =  { 2 };

    /* Number of features in training and testing data sets */
    private static final int nFeatures     = 3;
    /* Number of classes */
    private static final int nClasses      = 5;
    /* Number of tree in decision forest classification model */
    private static final int nTrees = 3;
    private static final int minObservationsInLeafNode = 8;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        /* Train DAAL Decision Forest Classification model */
        TrainingResult trainingResult = trainModel();
        System.out.println("Predict on trained model");
        double trainedAccurcy = testModel(trainingResult.get(TrainingResultId.model));

        /* Traverse the trained model to get Tree representation */
        Tree [] trees = traverseModel(trainingResult.get(TrainingResultId.model));
        /* Build the model by ModelBuilder from Tree */
        Model builtModel = buildModel(trees);
        System.out.println("Predict on built model from input user Tree ");
        double buildModelAccurcy = testModel(builtModel);

        if(trainedAccurcy == buildModelAccurcy) {
            System.out.println("Model was built successfully");
        }
        else {
            System.out.println("Model was built not correctly");
        }

        context.dispose();
    }

    public static Model buildModel(Tree [] trees) {

        /* Create a model builder */
        ModelBuilder modelBuilder = new ModelBuilder(context, nClasses, nTrees);

        /* Map to get relationship between nodes */
        Map<Node, ParentPlace> parentMap = new HashMap<Node, ParentPlace>();
        for(int i = 0; i < nTrees; i++) {
            final long nNodes = trees[i].nNodes;

            /* Allocate the memory for certain tree */
            modelBuilder.createTree(nNodes);
            boolean isRoot = true;

            /* Recursive DFS traversing of certain tree with building model */
            buildTree(i, trees[i].root, isRoot, modelBuilder, parentMap);
            parentMap.clear();
        }

        return modelBuilder.getModel();
    }

    private static boolean buildTree(long treeId, Node node, boolean isRoot, ModelBuilder builder, Map<Node, ParentPlace> parentMap) {

        if(node.left != null && node.right != null) {
            if(isRoot) {
                long parent = builder.addSplitNode(treeId, ModelBuilder.noParent, 0, node.featureIndex, node.featureValue);

                parentMap.put(node.left, new ParentPlace(parent,0));
                parentMap.put(node.right, new ParentPlace(parent,1));
                isRoot = false;
            }
            else {
                ParentPlace p = parentMap.get(node);
                long parent = builder.addSplitNode(treeId, p.parentId, p.place, node.featureIndex, node.featureValue);

                parentMap.put(node.left, new ParentPlace(parent,0));
                parentMap.put(node.right, new ParentPlace(parent,1));
            }
        }
        else {
            if(isRoot) {
                builder.addLeafNode(treeId, ModelBuilder.noParent, 0, node.classLabel);
                isRoot = false;
            }
            else {
                ParentPlace p = parentMap.get(node);
                builder.addLeafNode(treeId, p.parentId, p.place, node.classLabel);
            }
            return true;
        }
        buildTree(treeId, node.left, isRoot, builder, parentMap);
        buildTree(treeId, node.right, isRoot, builder, parentMap);
        return true;
    }

    private static TrainingResult trainModel() {
        /* Retrieve the data from the input data sets */
        FileDataSource trainDataSource = new FileDataSource(context, trainDatasetFileName,
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

        /* Pass a training data set and dependent values to the algorithm */
        algorithm.input.set(InputId.data, trainData);
        algorithm.input.set(InputId.labels, trainGroundTruth);

        /* Train the decision forest classification model */
        return algorithm.compute();
    }

    private static double testModel(Model model) {

        /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
        FileDataSource testDataSource = new FileDataSource(context, testDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for testing data and labels */
        NumericTable testData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.NotAllocate);
        NumericTable testGroundTruth = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.NotAllocate);
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
        PredictionResult predictionResult = algorithm.compute();
        NumericTable predictionResults = predictionResult.get(PredictionResultId.prediction);
        Service.printNumericTable("Decision forest prediction results (first 10 rows):", predictionResults, 10);
        Service.printNumericTable("Ground truth (first 10 rows):", testGroundTruth, 10);

        final long nRows = predictionResults.getNumberOfRows();
        long countOfNotCorrect = 0;
        for(long i = 0; i < nRows; i++) {
            if(predictionResults.getFloatValue(0,i) != testGroundTruth.getFloatValue(0,i)) {
                countOfNotCorrect++;
            }
        }
        double accuracy = 1 - (double)(countOfNotCorrect)/nRows;
        System.out.println("Accuracy: " + accuracy);

        return accuracy;
    }

    private static Tree [] traverseModel(Model model) {
        final int nTrees = (int)model.getNumberOfTrees();
        BFSNodeVisitor visitor = new BFSNodeVisitor(nTrees);
        for(int i = 0; i < nTrees; ++i) {
            model.traverseBFS(i, visitor);
        }
        return visitor.roots;
    }

}
