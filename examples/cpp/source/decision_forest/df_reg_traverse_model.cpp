/* file: df_reg_traverse_model.cpp */
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
!    C++ example of decision forest regression model traversal.
!
!    The program trains the decision forest regression model on a training
!    datasetFileName and prints the trained model by its depth-first traversing.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DF_REG_TRAVERSE_MODEL"></a>
 * \example df_reg_traverse_model.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms::decision_forest::regression;

/* Input data set parameters */
const string trainDatasetFileName = "../data/batch/df_regression_train.csv";
const size_t categoricalFeaturesIndices[] = { 3 };
const size_t nFeatures = 13;  /* Number of features in training and testing data sets */

/* Decision forest parameters */
const size_t nTrees = 2;

training::ResultPtr trainModel();
void loadData(const std::string& fileName, NumericTablePtr& pData, NumericTablePtr& pDependentVar);
void printModel(const daal::algorithms::decision_forest::regression::Model& m);

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &trainDatasetFileName);

    training::ResultPtr trainingResult = trainModel();
    printModel(*trainingResult->get(training::model));

    return 0;
}

training::ResultPtr trainModel()
{
    /* Create Numeric Tables for training data and dependent variables */
    NumericTablePtr trainData;
    NumericTablePtr trainDependentVariable;

    loadData(trainDatasetFileName, trainData, trainDependentVariable);

    /* Create an algorithm object to train the decision forest regression model with the default method */
    training::Batch<> algorithm;

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(training::data, trainData);
    algorithm.input.set(training::dependentVariable, trainDependentVariable);

    algorithm.parameter.nTrees = nTrees;

    /* Build the decision forest regression model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    return algorithm.getResult();
}

void loadData(const std::string& fileName, NumericTablePtr& pData, NumericTablePtr& pDependentVar)
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(fileName,
        DataSource::notAllocateNumericTable,
        DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and dependent variables */
    pData.reset(new HomogenNumericTable<>(nFeatures, 0, NumericTable::notAllocate));
    pDependentVar.reset(new HomogenNumericTable<>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(pData, pDependentVar));

    /* Retrieve the data from input file */
    trainDataSource.loadDataBlock(mergedData.get());

    NumericTableDictionaryPtr pDictionary = pData->getDictionarySharedPtr();
    for(size_t i = 0, n = sizeof(categoricalFeaturesIndices) / sizeof(categoricalFeaturesIndices[0]); i < n; ++i)
        (*pDictionary)[categoricalFeaturesIndices[i]].featureType = data_feature_utils::DAAL_CATEGORICAL;
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

void printModel(const daal::algorithms::decision_forest::regression::Model& m)
{
    PrintNodeVisitor visitor;
    std::cout << "Number of trees: " << m.getNumberOfTrees() << std::endl;
    for(size_t i = 0, n = m.getNumberOfTrees(); i < n; ++i)
    {
        std::cout << "Tree #" << i << std::endl;
        m.traverseDFS(i, visitor);
    }
}
