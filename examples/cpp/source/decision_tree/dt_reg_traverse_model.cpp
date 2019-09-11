/* file: dt_reg_traverse_model.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
!  Content:
!    C++ example of decision tree classification model traversal.
!
!    The program trains the decision tree classification model on a training
!    datasetFileName and prints the trained model by its depth-first traversing.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DT_REG_TRAVERSE_MODEL"></a>
 * \example dt_reg_traverse_model.cpp
 */

#include "daal.h"
#include "service.h"
#include <cstdio>

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string trainDatasetFileName            = "../data/batch/decision_tree_train.csv";
string pruneDatasetFileName            = "../data/batch/decision_tree_prune.csv";

const size_t nFeatures = 5; /* Number of features in training and testing data sets */

decision_tree::regression::training::ResultPtr trainModel();
void printModel(const daal::algorithms::decision_tree::regression::Model& m);

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 2, &trainDatasetFileName, &pruneDatasetFileName);

    decision_tree::regression::training::ResultPtr trainingResult = trainModel();
    printModel(*trainingResult->get(decision_tree::regression::training::model));

    return 0;
}

decision_tree::regression::training::ResultPtr trainModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName,
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and dependent variables */
    NumericTablePtr trainData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::notAllocate));
    NumericTablePtr trainGroundTruth(new HomogenNumericTable<>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(trainData, trainGroundTruth));

    /* Retrieve the data from the input file */
    trainDataSource.loadDataBlock(mergedData.get());

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the pruning input data from a .csv file */
    FileDataSource<CSVFeatureManager> pruneDataSource(pruneDatasetFileName,
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for pruning data and dependent variables */
    NumericTablePtr pruneData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::notAllocate));
    NumericTablePtr pruneGroundTruth(new HomogenNumericTable<>(1, 0, NumericTable::notAllocate));
    NumericTablePtr pruneMergedData(new MergedNumericTable(pruneData, pruneGroundTruth));

    /* Retrieve the data from the pruning input file */
    pruneDataSource.loadDataBlock(pruneMergedData.get());

    /* Create an algorithm object to train the Decision tree model */
    decision_tree::regression::training::Batch<> algorithm;

    /* Pass the training data set, dependent variables, and pruning dataset with dependent variables to the algorithm */
    algorithm.input.set(decision_tree::regression::training::data, trainData);
    algorithm.input.set(decision_tree::regression::training::dependentVariables, trainGroundTruth);
    algorithm.input.set(decision_tree::regression::training::dataForPruning, pruneData);
    algorithm.input.set(decision_tree::regression::training::dependentVariablesForPruning, pruneGroundTruth);

    /* Train the Decision tree model */
    algorithm.compute();

    /* Retrieve the results of the training algorithm  */
    return algorithm.getResult();
}

/** Visitor class implementing TreeNodeVisitor interface, prints out tree nodes of the model when it is called back by model traversal method */
class PrintNodeVisitor : public daal::algorithms::tree_utils::regression::TreeNodeVisitor
{
public:
    virtual bool onLeafNode(const daal::algorithms::tree_utils::regression::LeafNodeDescriptor &desc)
    {
        for(size_t i = 0; i < desc.level; ++i)
            std::cout << "  ";
        std::cout << "Level " << desc.level << ", leaf node. Response value = " << desc.response << ", Impurity = " << desc.impurity <<
            ", Number of samples = " << desc.nNodeSampleCount << std::endl;
        return true;
    }

    virtual bool onSplitNode(const daal::algorithms::tree_utils::regression::SplitNodeDescriptor &desc)
    {
        for(size_t i = 0; i < desc.level; ++i)
            std::cout << "  ";
        std::cout << "Level " << desc.level << ", split node. Feature index = " << desc.featureIndex <<
            ", feature value = " << desc.featureValue << ", Impurity = " << desc.impurity <<
            ", Number of samples = " << desc.nNodeSampleCount << std::endl;
        return true;
    }
};

void printModel(const daal::algorithms::decision_tree::regression::Model& m)
{
    PrintNodeVisitor visitor;
    m.traverseDFS(visitor);
}
