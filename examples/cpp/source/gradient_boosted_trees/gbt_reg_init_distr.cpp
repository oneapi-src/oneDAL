/* file: gbt_reg_init_distr.cpp */
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
!    C++ example of gradient boosted trees regression in the distributed processing mode.
!
!    The program trains the gradient boosted trees regression model on a training
!    datasetFileName and computes regression for the test data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-GBT_REG_DENSE_DISTR"></a>
 * \example gbt_reg_init_distr.cpp
 */
#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

typedef float FPType;

const string trainDataFileNames[] =
{
    "../data/distributed/df_regression_train_1.csv", "../data/distributed/df_regression_train_2.csv",
    "../data/distributed/df_regression_train_3.csv", "../data/distributed/df_regression_train_4.csv"
};

const size_t categoricalFeaturesIndices[] = { 3 };
const size_t nFeatures = 13;
const size_t nBlocks = 4;
const size_t maxBins = 4;
const size_t minBinSize = 5;

void loadData(const std::string& fileName, NumericTablePtr& pData, NumericTablePtr& pDependentVar);

int main(int argc, char *argv[])
{
    NumericTablePtr trainData[nBlocks];
    NumericTablePtr testData[nBlocks];
    NumericTablePtr trainDepVars[nBlocks];
    NumericTablePtr testDepVars[nBlocks];

    gbt::regression::init::Distributed<step2Master, FPType, gbt::regression::init::defaultDense> step2Algorithm(maxBins, minBinSize);

    for (size_t i = 0; i < 4; ++i)
    {
        loadData(trainDataFileNames[i], trainData[i], trainDepVars[i]);

        gbt::regression::init::Distributed<step1Local, FPType, gbt::regression::init::defaultDense> step1Algorithm(maxBins);

        step1Algorithm.input.set(gbt::regression::init::step1LocalData, trainData[i]);
        step1Algorithm.input.set(gbt::regression::init::step1LocalDependentVariables, trainDepVars[i]);

        step1Algorithm.compute();

        gbt::regression::init::DistributedPartialResultStep1Ptr step1Result = step1Algorithm.getPartialResult();

        printNumericTable(step1Result->get(gbt::regression::init::step1BinBorders), "Bin Borders");
        printNumericTable(step1Result->get(gbt::regression::init::step1BinSizes), "Bin Sizes");

        step2Algorithm.input.add(gbt::regression::init::step2MeanDependentVariable, step1Result->get(gbt::regression::init::step1MeanDependentVariable));
        step2Algorithm.input.add(gbt::regression::init::step2NumberOfRows, step1Result->get(gbt::regression::init::step1NumberOfRows));
        step2Algorithm.input.add(gbt::regression::init::step2BinBorders, step1Result->get(gbt::regression::init::step1BinBorders));
        step2Algorithm.input.add(gbt::regression::init::step2BinSizes, step1Result->get(gbt::regression::init::step1BinSizes));
    }

    step2Algorithm.compute();

    gbt::regression::init::DistributedPartialResultStep2Ptr step2Result = step2Algorithm.getPartialResult();

    printNumericTable(step2Result->get(gbt::regression::init::step2BinQuantities), "Bin Quantities");
    printNumericTable(step2Result->get(gbt::regression::init::step2MergedBinBorders), "Meged Bin Borders");

    for (size_t i = 0; i < 4; ++i)
    {
        gbt::regression::init::Distributed<step3Local, FPType, gbt::regression::init::defaultDense> step3Algorithm(maxBins);

        step3Algorithm.input.set(gbt::regression::init::step3MergedBinBorders, step2Result->get(gbt::regression::init::step2MergedBinBorders));
        step3Algorithm.input.set(gbt::regression::init::step3BinQuantities, step2Result->get(gbt::regression::init::step2BinQuantities));
        step3Algorithm.input.set(gbt::regression::init::step3LocalData, trainData[i]);
        step3Algorithm.input.set(gbt::regression::init::step3InitialResponse, step2Result->get(gbt::regression::init::step2InitialResponse));

        step3Algorithm.compute();

        printNumericTable(step3Algorithm.getPartialResult()->get(gbt::regression::init::step3BinnedData), "Binned Data");
    }

    return 0;
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
}
