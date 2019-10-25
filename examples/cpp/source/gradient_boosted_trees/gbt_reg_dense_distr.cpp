/* file: gbt_reg_dense_distr.cpp */
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
 * \example gbt_reg_dense_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::data_management;
using namespace daal::algorithms::gbt::regression;

typedef float algorithmFPType;

const size_t nBlocks = 4;

const string trainDatasetFileName[nBlocks] =
{
    "../data/distributed/df_regression_train_1.csv",
    "../data/distributed/df_regression_train_2.csv",
    "../data/distributed/df_regression_train_3.csv",
    "../data/distributed/df_regression_train_4.csv"
};

const string testDatasetFileName[nBlocks] =
{
    "../data/distributed/df_regression_test_1.csv",
    "../data/distributed/df_regression_test_2.csv",
    "../data/distributed/df_regression_test_3.csv",
    "../data/distributed/df_regression_test_4.csv"
};

const size_t categoricalFeaturesIndices[] = { 3 };
const size_t nFeatures = 13;
const size_t maxBins = 66000;
const size_t minBinSize = 5;
const size_t maxIterations = 40;

NumericTablePtr trainData[nBlocks];
NumericTablePtr trainDependentVariable[nBlocks];
NumericTablePtr testData[nBlocks];
NumericTablePtr testGroundTruth[nBlocks];

NumericTablePtr   binnedData[nBlocks];
NumericTablePtr   transposedBinnedData[nBlocks];
NumericTablePtr   dependentVariable[nBlocks];
NumericTablePtr   initialResponse[nBlocks];
NumericTablePtr   binSizes[nBlocks];
DataCollectionPtr binValues[nBlocks];
NumericTablePtr   treeStructure[nBlocks];
NumericTablePtr   treeOrder[nBlocks];
NumericTablePtr   optCoeffs[nBlocks];
NumericTablePtr   response[nBlocks];

DataCollectionPtr histograms[nBlocks];
DataCollectionPtr parentHistograms[nBlocks];
DataCollectionPtr totalHistograms[nFeatures];
DataCollectionPtr parentTotalHistograms[nFeatures];
DataCollectionPtr bestSplits;

DataCollectionPtr finalizedTrees[nBlocks];

ModelPtr partialModel[nBlocks];

int computeFinishedFlag(NumericTablePtr treeStructure);
void loadData(const std::string& fileName, NumericTablePtr& pData, NumericTablePtr& pDependentVar);
void initModel();
void trainModel();
void testModel();

int main(int argc, char *argv[])
{
    for (size_t i = 0; i < nBlocks; i++)
    {
        loadData(trainDatasetFileName[i], trainData[i], trainDependentVariable[i]);
    }

    initModel();

    trainModel();

    for (size_t i = 0; i < nBlocks; i++)
    {
        loadData(testDatasetFileName[i], testData[i], testGroundTruth[i]);
    }

    testModel();

    return 0;

} // end main

void initModel()
{
    init::Distributed<step2Master, algorithmFPType, init::defaultDense> step2Algorithm(maxBins, minBinSize);

    for (size_t i = 0; i < nBlocks; ++i)
    {
        init::Distributed<step1Local, algorithmFPType, init::defaultDense> step1Algorithm(maxBins);

        step1Algorithm.input.set(init::step1LocalData, trainData[i]);
        step1Algorithm.input.set(init::step1LocalDependentVariables, trainDependentVariable[i]);

        step1Algorithm.compute();

        init::DistributedPartialResultStep1Ptr step1Result = step1Algorithm.getPartialResult();

        step2Algorithm.input.add(init::step2MeanDependentVariable, step1Result->get(init::step1MeanDependentVariable));
        step2Algorithm.input.add(init::step2NumberOfRows, step1Result->get(init::step1NumberOfRows));
        step2Algorithm.input.add(init::step2BinBorders, step1Result->get(init::step1BinBorders));
        step2Algorithm.input.add(init::step2BinSizes, step1Result->get(init::step1BinSizes));
    }

    step2Algorithm.compute();

    init::DistributedPartialResultStep2Ptr step2Result = step2Algorithm.getPartialResult();

    for (size_t i = 0; i < nBlocks; i++)
    {
        initialResponse[i] = step2Result->get(init::step2InitialResponse);
        binValues[i] = step2Result->get(init::step2BinValues);
        binSizes[i] = step2Result->get(init::step2BinQuantities);
    }

    for (size_t i = 0; i < nBlocks; ++i)
    {
        init::Distributed<step3Local, algorithmFPType, init::defaultDense> step3Algorithm(maxBins);

        step3Algorithm.input.set(init::step3MergedBinBorders, step2Result->get(init::step2MergedBinBorders));
        step3Algorithm.input.set(init::step3BinQuantities, step2Result->get(init::step2BinQuantities));
        step3Algorithm.input.set(init::step3LocalData, trainData[i]);
        step3Algorithm.input.set(init::step3InitialResponse, step2Result->get(init::step2InitialResponse));

        step3Algorithm.compute();

        binnedData[i] = step3Algorithm.getPartialResult()->get(init::step3BinnedData);
        transposedBinnedData[i] = step3Algorithm.getPartialResult()->get(init::step3TransposedBinnedData);
        response[i] = step3Algorithm.getPartialResult()->get(init::step3Response);
        treeOrder[i] = step3Algorithm.getPartialResult()->get(init::step3TreeOrder);
    }
}

void trainModel()
{
    for (size_t i = 0; i < nBlocks; i++)
    {
        finalizedTrees[i] = DataCollectionPtr(new DataCollection());
    }
    for (size_t iter = 0; iter < maxIterations + 1; iter++)
    {
        // 1-st step "Update gradients local"
        for (size_t i = 0; i < nBlocks; i++)
        {
            training::Distributed<step1Local, algorithmFPType> step1;

            step1.input.set(training::step1BinnedData,         binnedData            [i]);
            step1.input.set(training::step1DependentVariable,  trainDependentVariable[i]);
            step1.input.set(training::step1InputResponse,      response              [i]);
            step1.input.set(training::step1InputTreeStructure, treeStructure         [i]);
            step1.input.set(training::step1InputTreeOrder,     treeOrder             [i]);

            step1.compute();

            response[i] = step1.getPartialResult()->get(training::response);
            optCoeffs[i] = step1.getPartialResult()->get(training::optCoeffs);
            treeOrder[i] = step1.getPartialResult()->get(training::treeOrder);
            treeStructure[i] = step1.getPartialResult()->get(training::step1TreeStructure);

            if (iter > 0)
            {
                finalizedTrees[i]->push_back(step1.getPartialResult()->get(training::finalizedTree));
            }
        }

        if (iter == maxIterations)
        {
            break;
        }

        for (size_t j = 0; j < nBlocks ; j++) parentHistograms[j] = DataCollectionPtr();
        for (size_t j = 0; j < nFeatures; j++) parentTotalHistograms[j] = DataCollectionPtr();

        while (!computeFinishedFlag(treeStructure[0])) // 2-nd step "Need to continue"
        {
            // 3-rd step "Compute histograms local"
            for (size_t i = 0; i < nBlocks; i++)
            {
                training::Distributed<step3Local, algorithmFPType> step3;

                step3.input.set(training::step3BinnedData,         binnedData      [i]);
                step3.input.set(training::step3BinSizes,           binSizes        [i]);
                step3.input.set(training::step3InputTreeStructure, treeStructure   [i]);
                step3.input.set(training::step3InputTreeOrder,     treeOrder       [i]);
                step3.input.set(training::step3OptCoeffs,          optCoeffs       [i]);

                step3.input.set(training::step3ParentHistograms,   parentHistograms[i]);

                step3.compute();
                histograms[i] = step3.getPartialResult()->get(training::histograms);

            }

            // 4-th step "Find best split local"

            DataCollectionPtr histogramsForFeature[nFeatures];

            for (size_t i = 0; i < nFeatures; i++)
            {
                histogramsForFeature[i] = DataCollectionPtr(new DataCollection(nBlocks));
                for (size_t j = 0; j < nBlocks; j++)
                {
                    DataCollectionPtr histogramsForBlocks(new DataCollection());
                    (*histogramsForFeature[i])[j] = histogramsForBlocks;
                    for (size_t k = 0; k < histograms[j]->size(); k++)
                    {
                        histogramsForBlocks->push_back(NumericTable::cast((*DataCollection::cast((*histograms[j])[k]))[i]));
                    }
                }
            }

            DataCollectionPtr bestSplits = DataCollectionPtr(new DataCollection());

            size_t processedFeatures = 0;

            for (size_t i = 0; i < nBlocks; i++)
            {
                size_t featuresForBlock = (nFeatures - processedFeatures) / (nBlocks - i);

                DataCollectionPtr featureIndices = DataCollectionPtr(new DataCollection());
                DataCollectionPtr parentTotalHistogramsForFeatures = DataCollectionPtr(new DataCollection());
                DataCollectionPtr partialHistogramsForFeatures = DataCollectionPtr(new DataCollection());

                for(size_t j = 0; j < featuresForBlock; j++ )
                {
                    featureIndices->push_back(HomogenNumericTable<int>::create(1, 1, NumericTableIface::doAllocate, (int)(processedFeatures + j)));
                    parentTotalHistogramsForFeatures->push_back(parentTotalHistograms[processedFeatures + j]);
                    partialHistogramsForFeatures->push_back(histogramsForFeature[processedFeatures + j]);
                }

                training::Distributed<step4Local, algorithmFPType> step4;
                step4.input.set(training::step4InputTreeStructure, treeStructure[0]);
                step4.input.set(training::step4FeatureIndices, featureIndices);
                step4.input.set(training::step4ParentTotalHistograms, parentTotalHistogramsForFeatures);
                step4.input.set(training::step4PartialHistograms, partialHistogramsForFeatures);

                step4.compute();

                DataCollectionPtr totalHistogramsForFeatures = step4.getPartialResult()->get(training::totalHistograms);
                DataCollectionPtr bestSplitsForFeatures = step4.getPartialResult()->get(training::bestSplits);

                for (size_t j = 0; j < featuresForBlock; j++)
                {
                    totalHistograms[processedFeatures + j] = DataCollection::cast((*totalHistogramsForFeatures)[j]);
                    bestSplits->push_back(DataCollection::cast((*bestSplitsForFeatures)[j]));
                }

                processedFeatures += featuresForBlock;
            }

            // 5-th step "Partition local"
            for (size_t i = 0; i < nBlocks; i++)
            {
                training::Distributed<step5Local, algorithmFPType> step5;

                step5.input.set(training::step5BinnedData,           binnedData          [i]);
                step5.input.set(training::step5TransposedBinnedData, transposedBinnedData[i]);
                step5.input.set(training::step5BinSizes,             binSizes            [i]);
                step5.input.set(training::step5InputTreeStructure,   treeStructure       [i]);
                step5.input.set(training::step5InputTreeOrder,       treeOrder           [i]);
                step5.input.set(training::step5PartialBestSplits,    bestSplits);

                step5.compute();
                treeStructure[i] = step5.getPartialResult()->get(training::step5TreeStructure);
                treeOrder[i] = step5.getPartialResult()->get(training::step5TreeOrder);
            }

            for (size_t j = 0; j < nBlocks; j++)
            {
                parentHistograms[j] = histograms[j];
            }
            for (size_t j = 0; j < nFeatures; j++)
            {
                parentTotalHistograms[j] = totalHistograms[j];
            }

        } // end while(NeedToContinue)

    } // end for(iter)


    for (size_t i = 0; i < nBlocks; i++)
    {
        training::Distributed<step6Local, algorithmFPType> step6;
        step6.input.set(training::step6InitialResponse, initialResponse[i]);
        step6.input.set(training::step6BinValues, binValues[i]);
        step6.input.set(training::step6FinalizedTrees, finalizedTrees[i]);

        step6.compute();

        partialModel[i] = step6.getPartialResult()->get(training::partialModel);
    }
}

int computeFinishedFlag(NumericTablePtr treeStructure)
{
    training::Distributed<step2Local, algorithmFPType> step2;
    step2.input.set(training::step2InputTreeStructure, treeStructure);
    step2.compute();
    NumericTablePtr res = step2.getPartialResult()->get(training::finishedFlag);
    return res->getValue<int>(0, 0);
}

void testModel()
{
    for (size_t i = 0; i < nBlocks; i++)
    {
        /* Create an algorithm object to predict values of gradient boosted trees regression */
        prediction::Batch<> algorithm;

        /* Pass a testing data set and the trained model to the algorithm */
        algorithm.input.set(prediction::data, testData[i]);
        algorithm.input.set(prediction::model, partialModel[i]);

        /* Predict values of gradient boosted trees regression */
        algorithm.compute();

        /* Retrieve the algorithm results */
        prediction::ResultPtr predictionResult = algorithm.getResult();

        printNumericTable(predictionResult->get(prediction::prediction),
            "Gragient boosted trees prediction results (first 10 rows):", 10);
        printNumericTable(testGroundTruth[i], "Ground truth (first 10 rows):", 10);
    }
}

void loadData(const std::string& fileName, NumericTablePtr& pData, NumericTablePtr& pDependentVar)
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(fileName,
        DataSource::notAllocateNumericTable,
        DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and dependent variables */
    pData.reset(new HomogenNumericTable<algorithmFPType>(nFeatures, 0, NumericTable::notAllocate));
    pDependentVar.reset(new HomogenNumericTable<algorithmFPType>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(pData, pDependentVar));

    /* Retrieve the data from input file */
    trainDataSource.loadDataBlock(mergedData.get());
}
