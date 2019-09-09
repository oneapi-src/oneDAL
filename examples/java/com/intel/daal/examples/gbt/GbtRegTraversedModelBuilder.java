/* file: GbtRegTraversedModelBuilder.java */
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
 //     Java example of gradient boosted trees regression model
 //     building from traversed gradient boosted trees regression model.
 //
 //     The program trains the gradient boosted trees regression model, gets
 //     pre-computed values from nodes of each tree using traversal and build
 //     model of the gradient boosted trees regression via Model Builder and
 //     computes regression for the test data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-GBTREGTRAVERSEDMODELBUILDER">
 * @example GbtRegTraversedModelBuilder.java
 */

package com.intel.daal.examples.gbt;

import com.intel.daal.algorithms.tree_utils.regression.TreeNodeVisitor;
import com.intel.daal.algorithms.tree_utils.regression.LeafNodeDescriptor;
import com.intel.daal.algorithms.tree_utils.SplitNodeDescriptor;
import com.intel.daal.algorithms.gbt.regression.training.InputId;
import com.intel.daal.algorithms.gbt.regression.training.TrainingResultId;
import com.intel.daal.algorithms.gbt.regression.prediction.ModelInputId;
import com.intel.daal.algorithms.gbt.regression.prediction.NumericTableInputId;
import com.intel.daal.algorithms.gbt.regression.prediction.PredictionResult;
import com.intel.daal.algorithms.gbt.regression.prediction.PredictionResultId;
import com.intel.daal.algorithms.gbt.regression.Model;
import com.intel.daal.algorithms.gbt.regression.ModelBuilder;
import com.intel.daal.algorithms.gbt.regression.prediction.*;
import com.intel.daal.algorithms.gbt.regression.training.*;
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


/* Node structure for representing nodes in trees after traversing DAAL model */
class NodeReg {
    public NodeReg left;
    public NodeReg right;
    public double response;
    public long featureIndex;
    public double featureValue;

    NodeReg(double rs, long fi, double fv) {
        left = null;
        right = null;
        response = rs;
        featureIndex = fi;
        featureValue = fv;
    }

    NodeReg() {
        left = null;
        right = null;
        response = 0;
        featureIndex = 0;
        featureValue = 0;
    }
}

/* Tree structure for representing tree after traversing DAAL model */
class TreeReg {
    public NodeReg root;
    public long nNodes;
}

/* Example of structure to remember relationship between nodes */
class ParentPlaceReg {
    public long parentId;
    public long place;

    ParentPlaceReg(long _parent, long _place) {
        parentId = _parent;
        place = _place;
    }

    ParentPlaceReg() {
        parentId = 0;
        place = 0;
    }
}

/* Visitor class implementing TreeNodeVisitor interface, prints out tree nodes of the model when it is called back by model traversal method */
class BFSNodeVisitorReg extends TreeNodeVisitor {

    public TreeReg [] roots;
    int treeId;
    Queue<NodeReg> parentNodes;

    BFSNodeVisitorReg(int nTrees) {
        roots = new TreeReg[nTrees];
        for(int i = 0; i < nTrees; i++) {
            roots[i] = new TreeReg();
            roots[i].root = new NodeReg();
        }
        treeId = 0;
        parentNodes = new LinkedList<NodeReg>();
    }

    @Override
    public boolean onLeafNode(LeafNodeDescriptor desc) {
        if(desc.level == 0) {
            NodeReg root = roots[treeId].root;
            roots[treeId].nNodes = 1;
            root.left = null;
            root.right = null;
            root.response = desc.response;
            root.featureIndex = 0;
            root.featureValue = 0;
            treeId++;
        }
        else {
            roots[treeId - 1].nNodes++;
            NodeReg node = new NodeReg(desc.response, 0, 0);

            NodeReg parent = parentNodes.peek();
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
            NodeReg root = roots[treeId].root;
            roots[treeId].nNodes = 1;
            root.left = null;
            root.right = null;
            root.response = 0;
            root.featureIndex = desc.featureIndex;
            root.featureValue = desc.featureValue;
            parentNodes.add(root);
            treeId++;
        }
        else {
            roots[treeId - 1].nNodes++;
            NodeReg node = new NodeReg(0,desc.featureIndex,desc.featureValue);

            NodeReg parent = parentNodes.peek();
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

class GbtRegTraversedModelBuilder {
    /* Input data set parameters */
    private static final String trainDatasetFileName = "../data/batch/df_regression_train.csv";
    private static final String testDatasetFileName = "../data/batch/df_regression_test.csv";
    private static final long categoricalFeaturesIndices [] =  { 3 };

    /* Number of features in training and testing data sets */
    private static final int nFeatures     = 13;
    /* Gradient boosted trees training parameters */
    private static int nTrees = 0;
    private static final int maxIterations = 40;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        /* Train DAAL Gradient Boosted Trees Regression model */
        TrainingResult trainingResult = trainModel();
        System.out.println("Predict on trained model");
        Model trainedModel = trainingResult.get(TrainingResultId.model);
        if(trainedModel != null) {
            nTrees = (int)trainedModel.getNumberOfTrees();
        }
        double trainedAccurcy = testModel(trainedModel);

        /* Traverse the trained model to get Tree representation */
        BFSNodeVisitorReg visitor = new BFSNodeVisitorReg(nTrees);
        TreeReg [] trees = traverseModel(trainedModel, visitor);
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

    public static Model buildModel(TreeReg [] trees) {

        /* Create a model builder */
        ModelBuilder modelBuilder = new ModelBuilder(context, nFeatures, nTrees);

        for(int i = 0; i < nTrees; i++) {
            final long nNodes = trees[i].nNodes;

            /* Allocate the memory for certain tree */
            modelBuilder.createTree(nNodes);
            boolean isRoot = true;

            /* Recursive DFS traversing of certain tree with building model */
            buildTree(i, trees[i].root, isRoot, modelBuilder, new ParentPlaceReg(0,0));
        }

        return modelBuilder.getModel();
    }

    private static boolean buildTree(long treeId, NodeReg node, boolean isRoot, ModelBuilder builder, ParentPlaceReg p) {

        if(node.left != null && node.right != null) {
            if(isRoot) {
                long parent = builder.addSplitNode(treeId, ModelBuilder.noParent, 0, node.featureIndex, node.featureValue);

                isRoot = false;
                buildTree(treeId, node.left, isRoot, builder, new ParentPlaceReg(parent, 0));
                buildTree(treeId, node.right, isRoot, builder, new ParentPlaceReg(parent, 1));
            }
            else {
                long parent = builder.addSplitNode(treeId, p.parentId, p.place, node.featureIndex, node.featureValue);

                buildTree(treeId, node.left, isRoot, builder, new ParentPlaceReg(parent, 0));
                buildTree(treeId, node.right, isRoot, builder, new ParentPlaceReg(parent, 1));
            }
        }
        else {
            if(isRoot) {
                builder.addLeafNode(treeId, ModelBuilder.noParent, 0, node.response);
                isRoot = false;
            }
            else {
                builder.addLeafNode(treeId, p.parentId, p.place, node.response);
            }
            return true;
        }

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
        trainData.getDictionary().setFeature(Float.class,3,DataFeatureUtils.FeatureType.DAAL_CATEGORICAL);

        /* Create algorithm objects to train the gradient boosted trees regression model */
        TrainingBatch algorithm = new TrainingBatch(context, Float.class, TrainingMethod.defaultDense);

        /* Pass a training data set and dependent values to the algorithm */
        algorithm.input.set(InputId.data, trainData);
        algorithm.input.set(InputId.dependentVariable, trainGroundTruth);
        algorithm.parameter.setMaxIterations(maxIterations);

        /* Train the decision forest regression model */
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
        testData.getDictionary().setFeature(Float.class, 3, DataFeatureUtils.FeatureType.DAAL_CATEGORICAL);

        /* Create algorithm objects for gradient boosted trees regression prediction */
        PredictionBatch algorithm = new PredictionBatch(context, Float.class, PredictionMethod.defaultDense);

        /* Pass a testing data set and the trained model to the algorithm */
        algorithm.input.set(NumericTableInputId.data, testData);
        algorithm.input.set(ModelInputId.model, model);

        /* Compute prediction results */
        PredictionResult predictionResult = algorithm.compute();
        NumericTable prediction = predictionResult.get(PredictionResultId.prediction);
        Service.printNumericTable("Gradient boosted trees prediction results (first 10 rows):", prediction, 10);
        Service.printNumericTable("Ground truth (first 10 rows):", testGroundTruth, 10);

        long nRows = 0;
        if(prediction != null) {
            nRows = prediction.getNumberOfRows();
        }

        double error = 0;
        for(long i = 0; i < nRows; i++) {
                error += (prediction.getFloatValue(0,i) - testGroundTruth.getFloatValue(0,i));
        }

        System.out.print("Error: ");
        System.out.format("%.3f%n", error);

        return error;
    }

    private static TreeReg [] traverseModel(Model model, BFSNodeVisitorReg visitor) {
        final int nTrees = (int)model.getNumberOfTrees();

        for(int i = 0; i < nTrees; ++i) {
            model.traverseBFS(i, visitor);
        }
        return visitor.roots;
    }

}
