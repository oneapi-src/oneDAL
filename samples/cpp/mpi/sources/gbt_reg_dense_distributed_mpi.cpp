/* file: gbt_reg_dense_distributed_mpi.cpp */
/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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
!    C++ example of the gradient boosting trees (GBT) regression
!    algorithm in the distributed processing mode.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-GBT_REG_DENSE_DISTRIBUTED"></a>
 * \example gbt_reg_dense_distributed_mpi.cpp
 */

#include "mpi.h"
#include "daal.h"
#include "service.h"
#include <cassert>
#include <iostream>

using namespace std;
using namespace daal;
using namespace daal::algorithms::gbt::regression;

const size_t nBlocks = 4;

const string trainDataFileNames[nBlocks] =
{
    "./data/distributed/df_regression_train_1.csv",
    "./data/distributed/df_regression_train_2.csv",
    "./data/distributed/df_regression_train_3.csv",
    "./data/distributed/df_regression_train_4.csv"
};

const string testDatasetFileName[nBlocks] =
{
    "./data/distributed/df_regression_test_1.csv",
    "./data/distributed/df_regression_test_2.csv",
    "./data/distributed/df_regression_test_3.csv",
    "./data/distributed/df_regression_test_4.csv"
};

const size_t nFeatures = 13;
const size_t localMaxBins = 256;
const size_t maxBins = 256;
const size_t minBinSize = 5;
const size_t maxIterations = 40;

typedef float algorithmFPType;

NumericTablePtr binnedData       ;
NumericTablePtr transposedBinnedData;
NumericTablePtr dependentVariable;
NumericTablePtr binSizes         ;
NumericTablePtr binOffsets       ;
NumericTablePtr treeStructure    ;
NumericTablePtr treeOrder        ;
NumericTablePtr optCoeffs        ;
NumericTablePtr binQuatities     ;
NumericTablePtr mergedBinBorders ;
NumericTablePtr response         ;
NumericTablePtr initialResponse  ;

NumericTablePtr trainData;
NumericTablePtr trainDependentVariable;
NumericTablePtr testData;
NumericTablePtr testGroundTruth;

DataCollectionPtr histograms      ;
DataCollectionPtr parentHistograms;
DataCollectionPtr finalizedTrees  ;
DataCollectionPtr binValues       ;

ModelPtr partialModel;

void sendCollectionMasterToAll(size_t beginId, size_t endId, size_t rankId,
                            size_t masterId, int tag, DataCollectionPtr & collection,
                            DataCollectionPtr & destCollection);
void sendCollectionAllToMaster(size_t beginId, size_t endId, size_t rankId,
                            size_t masterId, int tag, DataCollectionPtr & collection,
                            DataCollectionPtr & destCollection);
void sendTableAllToMaster(size_t beginId, size_t endId, size_t rankId,
                            size_t masterId, int tag, NumericTablePtr & table,
                            DataCollectionPtr & destCollection);
void sendTableMasterToAll(size_t beginId, size_t endId, size_t rankId,
                            size_t masterId, int tag, NumericTablePtr & table,
                            NumericTablePtr & destTable);
void sendCollection(DataCollectionPtr &collection, int recpnt, int tag);
void recvCollection(DataCollectionPtr &collection, int sender, int tag);
void sendRecvCollection(DataCollectionPtr &sendingCollection, int sender, DataCollectionPtr &destCollection, int recpnt, int tag);
void sendTable(NumericTablePtr &table, int recpnt, int tag);
void recvTable(NumericTablePtr &table, int sender, int tag);
void loadData(const std::string& fileName, NumericTablePtr& pData, NumericTablePtr& pDependentVar);

int computeFinishedFlag(NumericTablePtr treeStructure);

int rankId, comm_size;
#define mpi_root 0

void initModel();
void trainModel();
void testModel();

int commTag1 = 1;
int commTag2 = 2;

int main(int argc, char *argv[])
{
    int provided;

    MPI_Init_thread(&argc, &argv,MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);

    loadData(trainDataFileNames[rankId], trainData, trainDependentVariable);

    initModel();

    trainModel();

    loadData(testDatasetFileName[rankId], testData, testGroundTruth);

    testModel();

    MPI_Finalize();

    return 0;
}

void initModel()
{
    NumericTablePtr meanDependentVariable;
    NumericTablePtr numberOfRows;
    NumericTablePtr binBorders;
    NumericTablePtr binQuantities;
    NumericTablePtr mergedBinBorders;

    DataCollectionPtr meanDependentVariableCollection;
    DataCollectionPtr numberOfRowsCollection;
    DataCollectionPtr binBordersCollection;
    DataCollectionPtr binSizesCollection;

    NumericTablePtr _mergedBinBorders;
    NumericTablePtr _initialResponse;
    NumericTablePtr _binSizes;
    DataCollectionPtr _binValues;

    init::Distributed<step1Local, algorithmFPType, init::defaultDense> step1Algorithm(localMaxBins);

    step1Algorithm.input.set(init::step1LocalData, trainData);
    step1Algorithm.input.set(init::step1LocalDependentVariables, trainDependentVariable);

    step1Algorithm.compute();

    init::DistributedPartialResultStep1Ptr step1Result = step1Algorithm.getPartialResult();

    meanDependentVariableCollection = DataCollectionPtr(new DataCollection());
    numberOfRowsCollection = DataCollectionPtr(new DataCollection());
    binBordersCollection = DataCollectionPtr(new DataCollection());
    binSizesCollection = DataCollectionPtr(new DataCollection());

    meanDependentVariable = step1Result->get(init::step1MeanDependentVariable);
    numberOfRows = step1Result->get(init::step1NumberOfRows);
    binBorders = step1Result->get(init::step1BinBorders);
    binSizes = step1Result->get(init::step1BinSizes);

    sendTableAllToMaster(0,comm_size, rankId, mpi_root, 0, meanDependentVariable, meanDependentVariableCollection);
    sendTableAllToMaster(0,comm_size, rankId, mpi_root, 0, numberOfRows, numberOfRowsCollection);
    sendTableAllToMaster(0,comm_size, rankId, mpi_root, 0, binBorders, binBordersCollection);
    sendTableAllToMaster(0,comm_size, rankId, mpi_root, 0, binSizes, binSizesCollection);

    if (rankId == mpi_root)
    {
        init::Distributed<step2Master, algorithmFPType, init::defaultDense> step2Algorithm(maxBins, minBinSize);

        step2Algorithm.input.set(init::step2MeanDependentVariable, meanDependentVariableCollection);
        step2Algorithm.input.set(init::step2NumberOfRows, numberOfRowsCollection);
        step2Algorithm.input.set(init::step2BinBorders, binBordersCollection);
        step2Algorithm.input.set(init::step2BinSizes, binSizesCollection);

        step2Algorithm.compute();

        init::DistributedPartialResultStep2Ptr step2Result = step2Algorithm.getPartialResult();

        _initialResponse = step2Result->get(init::step2InitialResponse);
        _binValues = step2Result->get(init::step2BinValues);
        _binSizes = step2Result->get(init::step2BinQuantities);
        _mergedBinBorders = step2Result->get(init::step2MergedBinBorders);
    }

    sendTableMasterToAll(0, comm_size, rankId, mpi_root, 0, _initialResponse,  initialResponse );
    sendTableMasterToAll(0, comm_size, rankId, mpi_root, 0, _binSizes,         binSizes        );
    sendTableMasterToAll(0, comm_size, rankId, mpi_root, 0, _mergedBinBorders, mergedBinBorders);
    sendCollectionMasterToAll(0, comm_size, rankId, mpi_root, 0, _binValues,   binValues       );

    init::Distributed<step3Local, algorithmFPType, init::defaultDense> step3Algorithm(maxBins);

    step3Algorithm.input.set(init::step3MergedBinBorders, mergedBinBorders);
    step3Algorithm.input.set(init::step3BinQuantities, binSizes);
    step3Algorithm.input.set(init::step3LocalData, trainData);
    step3Algorithm.input.set(init::step3InitialResponse, initialResponse);

    step3Algorithm.compute();

    binnedData = step3Algorithm.getPartialResult()->get(init::step3BinnedData);
    transposedBinnedData = step3Algorithm.getPartialResult()->get(init::step3TransposedBinnedData);
    response   = step3Algorithm.getPartialResult()->get(init::step3Response);
    treeOrder  = step3Algorithm.getPartialResult()->get(init::step3TreeOrder);
}

void trainModel()
{
    size_t nFeaturesRation = (rankId < (nFeatures%comm_size)) + (nFeatures/comm_size); // count of Feature for current cluster node
    size_t nFeaturesRationPartner = 0;

    DataCollectionPtr totalHistograms[nFeaturesRation];
    DataCollectionPtr parentTotalHistograms[nFeaturesRation];

    DataCollectionPtr gatherHist[nFeaturesRation];// each element of array is collection of featurePacks for each cluster node
    DataCollectionPtr gatherHist1(new DataCollection());
    DataCollectionPtr destSplit(new DataCollection());

    finalizedTrees   = DataCollectionPtr(new DataCollection());
    parentHistograms = DataCollectionPtr(new DataCollection());

    int degreeForComm = 0;
            int cloneNRanks = comm_size;
            while(cloneNRanks > 0)
            {
                cloneNRanks = cloneNRanks >> 1;
                degreeForComm++;
            }
            int nShift = 1 << degreeForComm;

    for (size_t iter = 0; iter < maxIterations + 1; iter++)
    {
        // 1-st step "Update gradients local"
        training::Distributed<step1Local, algorithmFPType> step1;

        step1.input.set(training::step1BinnedData,        binnedData       );
        step1.input.set(training::step1DependentVariable, trainDependentVariable);
        step1.input.set(training::step1InputResponse,     response         );
        step1.input.set(training::step1InputTreeStructure,treeStructure    );
        step1.input.set(training::step1InputTreeOrder,    treeOrder        );

        step1.compute();

        response = step1.getPartialResult()->get(training::response);
        optCoeffs = step1.getPartialResult()->get(training::optCoeffs);
        treeOrder = step1.getPartialResult()->get(training::treeOrder);
        treeStructure = step1.getPartialResult()->get(training::step1TreeStructure);

        if (iter > 0)
        {
            finalizedTrees->push_back(step1.getPartialResult()->get(training::finalizedTree));
        }

        if (iter == maxIterations)
        {
            break;
        }

        parentHistograms = DataCollectionPtr();
        for (size_t j = 0; j < nFeaturesRation; j++) parentTotalHistograms[j] = DataCollectionPtr();

        while(!computeFinishedFlag(treeStructure)) // 2-nd step "Need to continue"
        {
            // 3-rd step "Compute histograms local"
            training::Distributed<step3Local, algorithmFPType> step3;

            step3.input.set(training::step3BinnedData,        binnedData   );
            step3.input.set(training::step3BinSizes,          binSizes     );
            step3.input.set(training::step3InputTreeStructure,treeStructure);
            step3.input.set(training::step3InputTreeOrder,    treeOrder    );
            step3.input.set(training::step3OptCoeffs,         optCoeffs    );

            step3.input.set(training::step3ParentHistograms,  parentHistograms);

            step3.compute();

            histograms = step3.getPartialResult()->get(training::histograms);

            // 4-th step "Find best split local"
            size_t nTreeNodes = histograms->size();

            DataCollectionPtr histForLocalFeatures(new DataCollection());
            DataCollectionPtr destCollection(new DataCollection());

            DataCollectionPtr partialHistogramsForFeatures = DataCollectionPtr(new DataCollection());
            size_t featureId = 0;
            gatherHist1 = DataCollectionPtr(new DataCollection());

            for (int shift = 0; shift < nShift; shift++)
            {
                int partner = rankId ^ shift;
                if (partner < comm_size)
                {
                    histForLocalFeatures = DataCollectionPtr(new DataCollection());
                    destCollection = DataCollectionPtr(new DataCollection());
                    size_t nFeaturesRationPartner = (partner < (nFeatures%comm_size)) + (nFeatures/comm_size);
                    for (size_t i = 0; i < nFeaturesRationPartner; i++)
                    {
                        DataCollectionPtr featurePack(new DataCollection());
                        featureId = partner + comm_size * i;
                        for (size_t j = 0; j < nTreeNodes; j++)
                        {
                            featurePack->push_back((*(DataCollection::cast((*histograms)[j])))[featureId]);
                        }
                        histForLocalFeatures->push_back(featurePack);
                    }
                    if (rankId == partner)
                    {
                        gatherHist1->push_back(histForLocalFeatures);
                    }
                    else
                    {
                        sendRecvCollection(histForLocalFeatures, rankId, destCollection, partner, commTag1);
                        gatherHist1->push_back(destCollection);
                    }
                }
            }

            for (size_t i = 0; i < nFeaturesRation; i++)
            {
                DataCollectionPtr histForOneFeature(new DataCollection());
                for (size_t j = 0; j < comm_size; j++)
                {
                    histForOneFeature->push_back(DataCollection::cast((*(DataCollection::cast((*gatherHist1)[j])))[i]));
                }
                partialHistogramsForFeatures->push_back(histForOneFeature);
            }

            DataCollectionPtr bestLocalSplits = DataCollectionPtr(new DataCollection());
            DataCollectionPtr featureIndices = DataCollectionPtr(new DataCollection());
            DataCollectionPtr parentTotalHistogramsForFeatures = DataCollectionPtr(new DataCollection());

            for(size_t i = 0; i < nFeaturesRation; i++ )
            {
                featureIndices->push_back(HomogenNumericTable<int>::create(1, 1, NumericTableIface::doAllocate, (int)(rankId + i*comm_size)));
                parentTotalHistogramsForFeatures->push_back(parentTotalHistograms[i]);
            }

            training::Distributed<step4Local, algorithmFPType> step4;
            step4.input.set(training::step4InputTreeStructure, treeStructure);
            step4.input.set(training::step4FeatureIndices, featureIndices);
            step4.input.set(training::step4ParentTotalHistograms, parentTotalHistogramsForFeatures);
            step4.input.set(training::step4PartialHistograms, partialHistogramsForFeatures);

            step4.compute();

            DataCollectionPtr totalHistogramsForFeatures = step4.getPartialResult()->get(training::totalHistograms);
            bestLocalSplits = step4.getPartialResult()->get(training::bestSplits);

            for (size_t i = 0; i < nFeaturesRation; i++)
            {
                totalHistograms[i] = DataCollection::cast((*totalHistogramsForFeatures)[i]);
            }

            // 5-th step "Partition local"

           DataCollectionPtr bestSplits = DataCollectionPtr(new DataCollection());

            for (int shift = 0; shift < nShift; shift++)
            {
                int partner = rankId ^ shift;
                if (partner < comm_size)
                {
                    destSplit = DataCollectionPtr(new DataCollection());
                    size_t nFeaturesRationPartner = (partner < (nFeatures%comm_size)) + (nFeatures/comm_size);

                    if (rankId == partner)
                    {
                        for (size_t f = 0; f < nFeaturesRationPartner; f++)
                        {
                            bestSplits->push_back(DataCollection::cast((*bestLocalSplits)[f]));
                        }
                    }
                    else
                    {
                        sendRecvCollection(bestLocalSplits, rankId, destSplit, partner, commTag2);

                        for (size_t f = 0; f < nFeaturesRationPartner; f++)
                        {
                            bestSplits->push_back(DataCollection::cast((*destSplit)[f]));
                        }
                    }
                }
            }

            training::Distributed<step5Local, algorithmFPType> step5;

            step5.input.set(training::step5BinnedData,          binnedData          );
            step5.input.set(training::step5TransposedBinnedData,transposedBinnedData);
            step5.input.set(training::step5BinSizes,            binSizes            );
            step5.input.set(training::step5InputTreeStructure,  treeStructure       );
            step5.input.set(training::step5InputTreeOrder,      treeOrder           );
            step5.input.set(training::step5PartialBestSplits,   bestSplits          );

            step5.compute();

            treeStructure = step5.getPartialResult()->get(training::step5TreeStructure);
            treeOrder     = step5.getPartialResult()->get(training::step5TreeOrder    );

            parentHistograms = histograms;

            for (size_t j = 0; j < nFeaturesRation; j++)
                parentTotalHistograms[j] = totalHistograms[j];
        } // end while(NeedToContinue)
    } // end for(iter)

    training::Distributed<step6Local, algorithmFPType> step6;

    step6.input.set(training::step6InitialResponse, initialResponse);
    step6.input.set(training::step6BinValues,       binValues      );
    step6.input.set(training::step6FinalizedTrees,  finalizedTrees );

    step6.compute();

    partialModel = step6.getPartialResult()->get(training::partialModel);
}

int computeFinishedFlag(NumericTablePtr treeStructure)
{

    training::Distributed<step2Local, algorithmFPType> step_2;
    step_2.input.set(training::step2InputTreeStructure, treeStructure);

    step_2.compute();

    NumericTablePtr res = step_2.getPartialResult()->get(training::finishedFlag);
    return res->getValue<int>(0,0);
}

void sendCollection(DataCollectionPtr &collection, int recpnt, int tag)
{
    ByteBuffer buff;
    size_t size = serializeDAALObject(collection.get(), buff);

    MPI_Send(&size, sizeof(size_t), MPI_BYTE, recpnt, tag * 2 + 0, MPI_COMM_WORLD);
    if(size)
    {
        MPI_Send(&buff[0], size, MPI_BYTE, recpnt, tag * 2 + 1, MPI_COMM_WORLD);
    }
}

void recvCollection(DataCollectionPtr &collection, int sender, int tag)
{
    size_t size = 0;
    MPI_Status status;
    MPI_Recv(&size, sizeof(size_t), MPI_BYTE, sender, tag * 2 + 0, MPI_COMM_WORLD, &status);
    if(size)
    {
        ByteBuffer buff(size);
        MPI_Recv(&buff[0], size, MPI_BYTE, sender, tag * 2 + 1, MPI_COMM_WORLD, &status);
        collection = DataCollection::cast(deserializeDAALObject(&buff[0], size));
    }
}

void sendRecvCollection(DataCollectionPtr &sendingCollection, int sender, DataCollectionPtr &destCollection, int recpnt, int tag)
{
    ByteBuffer sendingBuff;
    MPI_Status status;
    size_t sendingSize = serializeDAALObject(sendingCollection.get(), sendingBuff);
    size_t destSize = 0;
    if (sendingSize)
        MPI_Sendrecv(&sendingSize, sizeof(size_t) , MPI_BYTE, recpnt, 0,
            &destSize, sizeof(size_t), MPI_BYTE, recpnt, 0, MPI_COMM_WORLD, &status);
    if (destSize)
    {
        ByteBuffer destBuff(destSize);
        MPI_Sendrecv(&sendingBuff[0], sendingSize, MPI_BYTE, recpnt, 0,
            &destBuff[0], destSize, MPI_BYTE, recpnt, 0, MPI_COMM_WORLD, &status);
            destCollection = DataCollection::cast(deserializeDAALObject(&destBuff[0], destSize));
    }
}

void sendCollectionAllToMaster(size_t beginId, size_t endId, size_t rankId, size_t masterId, int tag, DataCollectionPtr & collection, DataCollectionPtr & destCollection)
{
    if (rankId == masterId)
    {
        for (size_t partnerId = beginId; partnerId < endId; partnerId++)
        {
            DataCollectionPtr partnerCollection;
            if (partnerId == rankId)
            {
                partnerCollection = collection;
            }
            else
            {
                recvCollection(partnerCollection, partnerId, tag);
            }

            if (partnerCollection.get())
            {
                destCollection->push_back(partnerCollection);
            }
        }
    }
    else
    {
        sendCollection(collection, masterId, tag);
    }
}

void sendCollectionMasterToAll(size_t beginId, size_t endId, size_t rankId, size_t masterId, int tag, DataCollectionPtr & collection, DataCollectionPtr & destCollection)
{
    if (rankId == masterId)
    {
        for (size_t partnerId = beginId; partnerId < endId; partnerId++)
        {
            if (partnerId == rankId)
            {
                destCollection = collection;
            }
            else
            {
                sendCollection(collection, partnerId, tag);
            }
        }
    }
    else
    {
        recvCollection(destCollection, masterId, tag);
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

void sendTable(NumericTablePtr & table, int recpnt, int tag)
{
    ByteBuffer buff;
    size_t size = (table.get() && table->getNumberOfRows() > 0) ? serializeDAALObject(table.get(), buff) : 0;
    MPI_Send(&size, sizeof(size_t), MPI_BYTE, recpnt, tag * 2 + 0, MPI_COMM_WORLD);
    if(size)
    {
        MPI_Send(&buff[0], size, MPI_BYTE, recpnt, tag * 2 + 1, MPI_COMM_WORLD);
    }
}

void recvTable(NumericTablePtr & table, int sender, int tag)
{
    size_t size = 0;
    MPI_Status status;
    MPI_Recv(&size, sizeof(size_t), MPI_BYTE, sender, tag * 2 + 0, MPI_COMM_WORLD, &status);
    if(size)
    {
        ByteBuffer buff(size);
        MPI_Recv(&buff[0], size, MPI_BYTE, sender, tag * 2 + 1, MPI_COMM_WORLD, &status);
        table = NumericTable::cast(deserializeDAALObject(&buff[0], size));
    }
}

void sendTableAllToMaster(size_t beginId, size_t endId, size_t rankId, size_t masterId, int tag, NumericTablePtr & table, DataCollectionPtr & destCollection)
{
    if (rankId == masterId)
    {
        for (size_t partnerId = beginId; partnerId < endId; partnerId++)
        {
            NumericTablePtr partnerTable;
            if (partnerId == rankId)
            {
                partnerTable = table;
            }
            else
            {
                recvTable(partnerTable, partnerId, tag);
            }

            if (partnerTable.get() && partnerTable->getNumberOfRows() > 0)
            {
                destCollection->push_back(partnerTable);
            }
        }
    }
    else
    {
        sendTable(table, masterId, tag);
    }
}

void sendTableMasterToAll(size_t beginId, size_t endId, size_t rankId, size_t masterId, int tag, NumericTablePtr & table, NumericTablePtr & destTable)
{
    if (rankId == masterId)
    {
        for (size_t partnerId = beginId; partnerId < endId; partnerId++)
        {
            if (partnerId == rankId)
            {
                destTable = table;
            }
            else
            {
                sendTable(table, partnerId, tag);
            }
        }
    }
    else
    {
        recvTable(destTable, masterId, tag);
    }
}

void testModel()
{
    NumericTablePtr prediction;

    /* Create an algorithm object to predict values of gradient boosted trees regression */
    prediction::Batch<> algorithm;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(prediction::data, testData);
    algorithm.input.set(prediction::model, partialModel);

    /* Predict values of gradient boosted trees regression */
    algorithm.compute();

    /* Retrieve the algorithm results */
    prediction::ResultPtr predictionResult = algorithm.getResult();

    for (int i = 0; i < comm_size; i++)
    {
        if (rankId == i)
        {
            printNumericTable(predictionResult->get(prediction::prediction),
                "Gradient Boosted Trees prediction results: (first 10 rows):", 10);
            printNumericTable(testGroundTruth, "Ground truth (first 10 rows):", 10);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}
