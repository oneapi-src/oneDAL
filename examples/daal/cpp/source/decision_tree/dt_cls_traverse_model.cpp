/* file: dt_cls_traverse_model.cpp */
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
!  Content:
!    C++ example of decision tree classification model traversal.
!
!    The program trains the decision tree classification model on a training
!    datasetFileName and prints the trained model by its depth-first traversing.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DT_CLS_TRAVERSE_MODEL"></a>
 * \example dt_cls_traverse_model.cpp
 */

#include "daal.h"
#include "service.h"
#include <cstdio>

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
std::string trainDatasetFileName = "../data/batch/decision_tree_train.csv";
std::string pruneDatasetFileName = "../data/batch/decision_tree_prune.csv";

const size_t nFeatures = 5; /* Number of features in training and testing data sets */
const size_t nClasses = 5; /* Number of classes */

decision_tree::classification::training::ResultPtr trainModel();
void printModel(const daal::algorithms::decision_tree::classification::Model& m);

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 2, &trainDatasetFileName, &pruneDatasetFileName);

    decision_tree::classification::training::ResultPtr trainingResult = trainModel();
    printModel(*trainingResult->get(classifier::training::model));

    return 0;
}

decision_tree::classification::training::ResultPtr trainModel() {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName,
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and labels */
    NumericTablePtr trainData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::notAllocate));
    NumericTablePtr trainGroundTruth(new HomogenNumericTable<>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(trainData, trainGroundTruth));

    /* Retrieve the data from the input file */
    trainDataSource.loadDataBlock(mergedData.get());

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the pruning input data from a .csv file */
    FileDataSource<CSVFeatureManager> pruneDataSource(pruneDatasetFileName,
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for pruning data and labels */
    NumericTablePtr pruneData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::notAllocate));
    NumericTablePtr pruneGroundTruth(new HomogenNumericTable<>(1, 0, NumericTable::notAllocate));
    NumericTablePtr pruneMergedData(new MergedNumericTable(pruneData, pruneGroundTruth));

    /* Retrieve the data from the pruning input file */
    pruneDataSource.loadDataBlock(pruneMergedData.get());

    /* Create an algorithm object to train the Decision tree model */
    decision_tree::classification::training::Batch<> algorithm(nClasses);

    /* Pass the training data set, labels, and pruning dataset with labels to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruth);
    algorithm.input.set(decision_tree::classification::training::dataForPruning, pruneData);
    algorithm.input.set(decision_tree::classification::training::labelsForPruning,
                        pruneGroundTruth);

    /* Train the Decision tree model */
    algorithm.compute();

    /* Retrieve the results of the training algorithm  */
    return algorithm.getResult();
}

/** Visitor class implementing TreeNodeVisitor interface, prints out tree nodes of the model when it is called back by model traversal method */
class PrintNodeVisitor : public daal::algorithms::tree_utils::classification::TreeNodeVisitor {
public:
    virtual bool onLeafNode(const tree_utils::classification::LeafNodeDescriptor& desc) {
        for (size_t i = 0; i < desc.level; ++i)
            std::cout << "  ";
        std::cout << "Level " << desc.level << ", leaf node. Response value = " << desc.label
                  << ", Impurity = " << desc.impurity
                  << ", Number of samples = " << desc.nNodeSampleCount << std::endl;
        return true;
    }

    virtual bool onSplitNode(const tree_utils::classification::SplitNodeDescriptor& desc) {
        for (size_t i = 0; i < desc.level; ++i)
            std::cout << "  ";
        std::cout << "Level " << desc.level << ", split node. Feature index = " << desc.featureIndex
                  << ", feature value = " << desc.featureValue << ", Impurity = " << desc.impurity
                  << ", Number of samples = " << desc.nNodeSampleCount << std::endl;
        return true;
    }
};

void printModel(const daal::algorithms::decision_tree::classification::Model& m) {
    PrintNodeVisitor visitor;
    m.traverseDFS(visitor);
}
