/* file: gbt_regression_train_dense_default_distr_impl.i */
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
//++
//  Implementation of auxiliary functions for gradient boosted trees regression
//  (defaultDense) method.
//--
*/

#ifndef __GBT_REGRESSION_TRAIN_DENSE_DEFAULT_DISTR_IMPL_I__
#define __GBT_REGRESSION_TRAIN_DENSE_DEFAULT_DISTR_IMPL_I__

#include "gbt_regression_train_kernel.h"
#include "gbt_regression_model_impl.h"
#include "gbt_regression_loss_impl.h"
#include "gbt_regression_tree_impl.h"
#include "gbt_train_dense_default_impl.i"
#include "gbt_train_tree_builder.i"
#include "gbt_train_hist_kernel.i"
#include "gbt_model_impl.h"

using namespace daal::algorithms::dtrees::training::internal;
using namespace daal::algorithms::gbt::training::internal;
using namespace daal::algorithms::gbt::regression::internal;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace training
{
namespace internal
{

//////////////////////////////////////////////////////////////////////////////////////////
// RegressionTrainDistrStep1Kernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, gbt::regression::training::Method method, CpuType cpu>
services::Status RegressionTrainDistrStep1Kernel<algorithmFPType, method, cpu>::compute(const NumericTable *ntBinnedData,
                                                                                        const NumericTable *ntDependentVariable,
                                                                                        const NumericTable *ntInputResponse,
                                                                                              NumericTable *ntInputTreeStructure,
                                                                                        const NumericTable *ntInputTreeOrder,
                                                                                              NumericTable *ntResponse,
                                                                                              NumericTable *ntOptCoeffs,
                                                                                              NumericTable *ntTreeOrder,
                                                                                              NumericTable *ntFinalizedTree,
                                                                                              NumericTable *ntTreeStructure,
                                                                                        const Parameter    &par)
{
    typedef LossFunction<algorithmFPType, cpu> LossFunctionType;
    typedef ImpurityData<algorithmFPType, cpu> ImpurityType;
    typedef TreeTableConnector<algorithmFPType> ConnectorType;

    const size_t nRows = ntBinnedData->getNumberOfRows();

    bool isFirstIteration = (ntInputTreeStructure == nullptr);
    bool isTreeStructureInplace = (ntInputTreeStructure == ntFinalizedTree);

    DAAL_ASSERT(isFirstIteration || isTreeStructureInplace);

    ReadRows<algorithmFPType, cpu> inputResponseRows(const_cast<NumericTable *>(ntInputResponse), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(inputResponseRows);
    const algorithmFPType * const inputResponse = inputResponseRows.get();

    WriteRows<algorithmFPType, cpu> responseRows(ntResponse, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(responseRows);
    algorithmFPType * const response = responseRows.get();

    if (!isFirstIteration)
    {
        // UPDATE RESPONSES

        ReadRows<int, cpu> inputTreeOrderRows(const_cast<NumericTable *>(ntInputTreeOrder), 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(inputTreeOrderRows);
        const int * const inputTreeOrder = inputTreeOrderRows.get();

        ConnectorType inputConnector(dynamic_cast<AOSNumericTable *>(ntInputTreeStructure));

        Collection<TableRecord<algorithmFPType> *> leaves;
        inputConnector.getLeafNodes(0, leaves);
        size_t nLeaves = leaves.size();

//        for (size_t t = 0; t < nLeaves; t++)
        LoopHelper<cpu>::run(true, nLeaves, [&](size_t t)
        {
            TableRecord<algorithmFPType> *node = leaves[t];

            const size_t curRows = node->n;
            const size_t curOffset = node->iStart;
            const ImpurityType imp(node->gTotal, node->hTotal);

            algorithmFPType res = 0;

            algorithmFPType val = imp.h + par.lambda;
            if(!isZero<algorithmFPType, cpu>(val))
            {
                val = -imp.g / val;
                const algorithmFPType inc = val * par.shrinkage;

                res = inc;

                const size_t blockSize = 2048;
                size_t nBlocks = curRows / blockSize;
                nBlocks += !!(curRows - nBlocks * blockSize);

                LoopHelper<cpu>::run(true, nBlocks, [&](size_t block)
                {
                    const size_t iStart = block * blockSize + curOffset;
                    const size_t iEnd = (((block + 1) * blockSize > curRows) ? curOffset + curRows : iStart + blockSize);

                    PRAGMA_IVDEP
                    PRAGMA_VECTOR_ALWAYS
                    for (size_t i = iStart; i < iEnd; i++)
                    {
                        response[inputTreeOrder[i]] = inputResponse[inputTreeOrder[i]] + inc;
                    }
                });
            }

            node->response = res;
            node->isFinalized = 1;
        });
    }
    else
    {
        if (ntInputResponse != ntResponse)
        {
            services::daal_memcpy_s(response, sizeof(algorithmFPType) * nRows, inputResponse, sizeof(algorithmFPType) * nRows);
        }
    }

    LossFunctionType *loss;

    switch(par.loss)
    {
        case squared:
            loss = new SquaredLoss<algorithmFPType, cpu>(); break;
        default:
            DAAL_ASSERT(false);
    }

    ReadRows<algorithmFPType, cpu> dependentVariableRows(const_cast<NumericTable *>(ntDependentVariable), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(dependentVariableRows);
    const algorithmFPType * const dependentVariable = dependentVariableRows.get();

    WriteRows<algorithmFPType, cpu> optCoeffsRows(ntOptCoeffs, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(optCoeffsRows);
    algorithmFPType * const optCoeffs = optCoeffsRows.get();

    WriteRows<int, cpu> treeOrderRows(const_cast<NumericTable *>(ntTreeOrder), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(treeOrderRows);
    int * const treeOrder = treeOrderRows.get();

    const size_t blockSize = 2048;
    size_t nBlocks = nRows / blockSize;
    nBlocks += !!(nRows - nBlocks * blockSize);

    LoopHelper<cpu>::run(true, nBlocks, [&](size_t block)
    {
        const size_t iStart = block * blockSize;
        const size_t iEnd = (((block + 1) * blockSize > nRows) ? nRows : iStart + blockSize);

        loss->getGradients(iEnd - iStart, iEnd - iStart, &(dependentVariable[iStart]), &(response[iStart]), nullptr, &(optCoeffs[2 * iStart]));

        for (size_t i = iStart; i < iEnd; i++)
        {
            treeOrder[i] = i;
        }
    });

    delete loss;

    ConnectorType connector(dynamic_cast<AOSNumericTable *>(ntTreeStructure));

    TableRecord<algorithmFPType> *record = connector.get(0);

    record->level = 0;
    record->nid = 0;
    record->iStart = 0;
    record->n = nRows;
    record->nodeState = ConnectorType::split;
    record->isFinalized = false;

    return services::Status();
}


//////////////////////////////////////////////////////////////////////////////////////////
// RegressionTrainDistrStep2Kernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, gbt::regression::training::Method method, CpuType cpu>
services::Status RegressionTrainDistrStep2Kernel<algorithmFPType, method, cpu>::compute(      NumericTable *ntInputTreeStructure,
                                                                                              NumericTable *ntFinishedFlag)
{
    typedef TreeTableConnector<algorithmFPType> ConnectorType;

    ConnectorType inputConnector(dynamic_cast<AOSNumericTable *>(ntInputTreeStructure));

    Collection<TableRecord<algorithmFPType> *> nodesForSplit;
    inputConnector.getSplitNodes(0, nodesForSplit);
    size_t nNodesForSplit = nodesForSplit.size();

    WriteRows<int, cpu> finishedFlagRows(ntFinishedFlag, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(finishedFlagRows);
    int * const finishedFlag = finishedFlagRows.get();
    finishedFlag[0] = int(nNodesForSplit == 0);

    return services::Status();
}

//////////////////////////////////////////////////////////////////////////////////////////
// RegressionTrainDistrStep3Kernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, gbt::regression::training::Method method, CpuType cpu>
services::Status RegressionTrainDistrStep3Kernel<algorithmFPType, method, cpu>::compute(const NumericTable   *ntBinnedData,
                                                                                        const NumericTable   *ntBinSizes,
                                                                                              NumericTable   *ntInputTreeStructure,
                                                                                        const NumericTable   *ntInputTreeOrder,
                                                                                        const NumericTable   *ntOptCoeffs,
                                                                                        const DataCollection *dcParentHistograms,
                                                                                              DataCollection *dcHistograms)
{
    const size_t nRows = ntBinnedData->getNumberOfRows();
    const size_t nFeatures = ntBinnedData->getNumberOfColumns();

    ReadRows<int, cpu> binSizesRows(const_cast<NumericTable *>(ntBinSizes), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(binSizesRows);
    const int * const binSizes = binSizesRows.get();

    int maxBinSize = 0;

    for (size_t i = 0; i < nFeatures; i++)
    {
        if (binSizes[i] > maxBinSize)
        {
            maxBinSize = binSizes[i];
        }
    }

    if (maxBinSize <= 256)
    {
        HomogenNumericTable<unsigned char> *ntBinnedData8 = dynamic_cast<HomogenNumericTable<unsigned char> *>(const_cast<NumericTable *>(ntBinnedData));
        if (ntBinnedData8)
        {
            const uint8_t *data = ntBinnedData8->getArray();
            return computeImpl<uint8_t>(data, ntBinSizes, ntInputTreeStructure, ntInputTreeOrder, ntOptCoeffs, dcParentHistograms, dcHistograms);
        }
    }

    if (maxBinSize <= 65536)
    {
        HomogenNumericTable<unsigned short> *ntBinnedData16 = dynamic_cast<HomogenNumericTable<unsigned short> *>(const_cast<NumericTable *>(ntBinnedData));
        if (ntBinnedData16)
        {
            const uint16_t *data = ntBinnedData16->getArray();
            return computeImpl<uint16_t>(data, ntBinSizes, ntInputTreeStructure, ntInputTreeOrder, ntOptCoeffs, dcParentHistograms, dcHistograms);

        }
    }

    ReadRows<int, cpu> ntBinnedDataRows(const_cast<NumericTable *>(const_cast<NumericTable *>(ntBinnedData)), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(ntBinnedDataRows);
    const int * const data = ntBinnedDataRows.get();

    return computeImpl<int>(data, ntBinSizes, ntInputTreeStructure, ntInputTreeOrder, ntOptCoeffs, dcParentHistograms, dcHistograms);
}

template <typename algorithmFPType, typename BinIndexType, CpuType cpu>
services::Status computeHistograms(const BinIndexType * const data,
                                   const int * const binSizes,
                                   const algorithmFPType * const optCoeffs,
                                   const size_t nRows,
                                   const size_t nFeatures,
                                   const size_t nDiffFeatMax,
                                   const int * const inputTreeOrder,
                                   GlobalStorages<algorithmFPType, BinIndexType, cpu> &storage,
                                   MergedResult<hist::Result<algorithmFPType, cpu>, cpu> *result,
                                   const TableRecord<algorithmFPType> *node)
{
    typedef ghSum<algorithmFPType, cpu> GHSumType;

    if (node->n == 0)
    {
        LoopHelper<cpu>::run(true, nFeatures, [&](size_t i)
        {
            hist::Result<algorithmFPType, cpu> &res = result->res[i];

            const size_t nUnique = binSizes[i];

            if (!res.ghSums)
            {
                res.ghSums = storage.singleGHSums.get(i).getBlockFromStorage();
            }
            res.gTotal = 0;
            res.hTotal = 0;
            res.iFeature = i;
            res.nUnique = nUnique;

            const size_t iStart = storage.nUniquesArr[i];
            const size_t iEnd = iStart + nUnique;

            hist::GHSumsHelper<algorithmFPType, int, BinIndexType, GHSumType, cpu>::fillByZero(nUnique, res.ghSums);
        });

        return services::Status();
    }

    const size_t blockSize = 2048;
    size_t nBlocks = node->n / blockSize;
    nBlocks += !!(node->n - nBlocks * blockSize);

    TlsGHSumMerge<GHSumForTLS<GHSumType, cpu>, algorithmFPType, cpu>* tls = storage.GHForCols.getBlockFromStorage();

    LoopHelper<cpu>::run(true, nBlocks, [&](size_t block)
    {
        const size_t iStart = block*blockSize + node->iStart;
        const size_t iEnd = (((block + 1) * blockSize > node->n) ? node->iStart + node->n : iStart + blockSize);

        auto* local = tls->local();
        GHSumType* aGHSum = local->ghSum;
        algorithmFPType* aGHSumFP = (algorithmFPType*)local->ghSum;

        if (!local->isInitialized)
        {
            hist::GHSumsHelper<algorithmFPType, int, BinIndexType, GHSumType, cpu>::fillByZero(nDiffFeatMax, aGHSum);
            local->isInitialized = true;
        }

        algorithmFPType* pgh = (algorithmFPType *)optCoeffs;
        hist::ComputeGHSumByRows<RowIndexType, BinIndexType, algorithmFPType, cpu>::run(aGHSumFP, data, inputTreeOrder, pgh, nFeatures, iStart, iEnd, node->iStart + node->n, storage.nUniquesArr.get());

//          DAAL_TYPENAME SplitMode::ComputeGHSumsTask task(block, blockSize, this->_data, this->node, tls);
//          task.execute();
    });

    algorithmFPType** ptrs = services::internal::service_scalable_malloc<algorithmFPType*, cpu>(nBlocks);
    size_t size;
    tls->reduceTo(ptrs, size);

    LoopHelper<cpu>::run(true, nFeatures, [&](size_t i)
    {
        hist::Result<algorithmFPType, cpu> &res = result->res[i];

        const size_t nUnique = binSizes[i];

        if (!res.ghSums)
        {
            res.ghSums = storage.singleGHSums.get(i).getBlockFromStorage();
        }
        res.gTotal = 0;
        res.hTotal = 0;
        res.iFeature = i;
        res.nUnique = nUnique;

        const size_t iStart = storage.nUniquesArr[i];
        const size_t iEnd = iStart + nUnique;

        hist::MergeGHSums<algorithmFPType, RowIndexType, BinIndexType, cpu>::run(nUnique, iStart, iEnd, ptrs, size, res);

//          DAAL_TYPENAME SplitMode::FindBestSplitTask task(i, nBlocks, this->_data, this->node, bestSplit,
//                      this->_result->res[i], ptrs, size);
//          task.execute();
    });

    tls->release();
    storage.GHForCols.returnBlockToStorage(tls);
    services::internal::service_scalable_free<algorithmFPType*, cpu>(ptrs);

    return services::Status();
}

template <typename algorithmFPType, gbt::regression::training::Method method, CpuType cpu>
template <typename BinIndexType>
services::Status RegressionTrainDistrStep3Kernel<algorithmFPType, method, cpu>::computeImpl(const BinIndexType * const data,
                                                                                            const NumericTable        *ntBinSizes,
                                                                                                  NumericTable        *ntInputTreeStructure,
                                                                                            const NumericTable        *ntInputTreeOrder,
                                                                                            const NumericTable        *ntOptCoeffs,
                                                                                            const DataCollection      *dcParentHistograms,
                                                                                                  DataCollection      *dcHistograms)
{
    typedef hist::Result<algorithmFPType, cpu> ResultType;
    typedef MergedResult<ResultType, cpu> MergedResultType;
    typedef ghSum<algorithmFPType, cpu> GHSumType;
    typedef TreeTableConnector<algorithmFPType> ConnectorType;

    const size_t nRows = ntOptCoeffs->getNumberOfRows();
    const size_t nFeatures = ntBinSizes->getNumberOfColumns();

    ReadRows<int, cpu> inputTreeOrderRows(const_cast<NumericTable *>(ntInputTreeOrder), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(inputTreeOrderRows);
    const int * const inputTreeOrder = inputTreeOrderRows.get();

    ReadRows<algorithmFPType, cpu> optCoeffsRows(const_cast<NumericTable *>(ntOptCoeffs), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(optCoeffsRows);
    const algorithmFPType * const optCoeffs = optCoeffsRows.get();

    ConnectorType inputConnector(dynamic_cast<AOSNumericTable *>(ntInputTreeStructure));

    Collection<SplitRecord<algorithmFPType> > nodesForSplit;
    inputConnector.getSplitNodesMerged(0, nodesForSplit);
    size_t nNodesForSplit = nodesForSplit.size();

    ReadRows<int, cpu> binSizesRows(const_cast<NumericTable *>(ntBinSizes), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(binSizesRows);
    const int * const binSizes = binSizesRows.get();

    TVector<size_t, cpu, ScalableAllocator<cpu> > nUniquesArr(nFeatures);
    DAAL_CHECK_MALLOC(nUniquesArr.get());
    size_t nDiffFeatMax = 0;
    nUniquesArr[0] = 0;
    for(size_t i = 0; i < nFeatures; ++i)
    {
        nUniquesArr[i] = nDiffFeatMax;
        nDiffFeatMax += binSizes[i];
    }

    GlobalStorages<algorithmFPType, BinIndexType, cpu> storage(nFeatures, nFeatures, nDiffFeatMax, 2);
    storage.nUniquesArr = nUniquesArr;
    storage.nDiffFeatMax = nDiffFeatMax;

    for(size_t i = 0; i < nFeatures; ++i)
    {
        storage.singleGHSums.add(i, binSizes[i], 2);
    }

    TArray<int, cpu> histogramsOffsets(nNodesForSplit);
    DAAL_CHECK_MALLOC(histogramsOffsets.get());

    size_t nHistograms = 0;

    for (size_t t = 0; t < nNodesForSplit; t++)
    {
        histogramsOffsets[t] = nHistograms;
        SplitRecord<algorithmFPType>& record = nodesForSplit[t];
        nHistograms += int (record.first != NULL) + int (record.second != NULL);
    }

    dcHistograms->clear();
    for(size_t i = 0; i < nHistograms; i++)
    {
        dcHistograms->push_back(DataCollectionPtr(new DataCollection(nFeatures)));
    }

//    for (size_t t = 0; t < nNodesForSplit; t++)
    LoopHelper<cpu>::run(true, nNodesForSplit, [&](size_t t)
    {
        SplitRecord<algorithmFPType>& record = nodesForSplit[t];

        if (record.first == NULL && record.second == NULL)
        {
            // continue;
        } else {
        DataCollectionPtr parentHistogramsForNode = dcParentHistograms ? DataCollection::cast((*dcParentHistograms)[t]) : DataCollectionPtr();

        if (record.first && record.second)
        {
            DataCollectionPtr histogramsForLeftNode = DataCollection::cast((*dcHistograms)[histogramsOffsets[t]]);
            DataCollectionPtr histogramsForRightNode = DataCollection::cast((*dcHistograms)[histogramsOffsets[t] + 1]);

            DAAL_ASSERT(parentHistogramsForNode.get());

            const TableRecord<algorithmFPType> *left  = record.first;
            const TableRecord<algorithmFPType> *right = record.second;

            const bool computeLeft = left->n < right->n;

            for(size_t i = 0; i < nFeatures; ++i)
            {
                services::Status s;
                services::SharedPtr<HomogenNumericTable<algorithmFPType> > ntHistogramLeft = HomogenNumericTable<algorithmFPType>::create(4, binSizes[i], NumericTable::doAllocate, &s);
//                DAAL_CHECK_STATUS_VAR(s);
                services::SharedPtr<HomogenNumericTable<algorithmFPType> > ntHistogramRight = HomogenNumericTable<algorithmFPType>::create(4, binSizes[i], NumericTable::doAllocate, &s);
//                DAAL_CHECK_STATUS_VAR(s);
                (*histogramsForLeftNode)[i] = ntHistogramLeft;
                (*histogramsForRightNode)[i] = ntHistogramRight;
            }

            MergedResult<ResultType, cpu> *resultCompute = new (services::internal::service_scalable_malloc<MergedResultType, cpu>(1)) MergedResultType(nFeatures);
            MergedResult<ResultType, cpu> *resultDiff = new (services::internal::service_scalable_malloc<MergedResultType, cpu>(1)) MergedResultType(nFeatures);

            if (computeLeft)
            {
                for(size_t i = 0; i < nFeatures; ++i)
                {
                    services::SharedPtr<HomogenNumericTable<algorithmFPType> > ntHistogramLeft = HomogenNumericTable<algorithmFPType>::cast((*histogramsForLeftNode)[i]);
                    services::SharedPtr<HomogenNumericTable<algorithmFPType> > ntHistogramRight = HomogenNumericTable<algorithmFPType>::cast((*histogramsForRightNode)[i]);
                    resultCompute->res[i].ghSums = (ghSum<algorithmFPType, cpu> *)ntHistogramLeft->getArray();
                    resultDiff->res[i].ghSums = (ghSum<algorithmFPType, cpu> *)ntHistogramRight->getArray();
                }

                computeHistograms<algorithmFPType, BinIndexType, cpu>(data, binSizes, optCoeffs, nRows, nFeatures, nDiffFeatMax, inputTreeOrder, storage, resultCompute, left);
            }
            else
            {
                for(size_t i = 0; i < nFeatures; ++i)
                {
                    services::SharedPtr<HomogenNumericTable<algorithmFPType> > ntHistogramLeft = HomogenNumericTable<algorithmFPType>::cast((*histogramsForLeftNode)[i]);
                    services::SharedPtr<HomogenNumericTable<algorithmFPType> > ntHistogramRight = HomogenNumericTable<algorithmFPType>::cast((*histogramsForRightNode)[i]);
                    resultCompute->res[i].ghSums = (ghSum<algorithmFPType, cpu> *)ntHistogramRight->getArray();
                    resultDiff->res[i].ghSums = (ghSum<algorithmFPType, cpu> *)ntHistogramLeft->getArray();
                }

                computeHistograms<algorithmFPType, BinIndexType, cpu>(data, binSizes, optCoeffs, nRows, nFeatures, nDiffFeatMax, inputTreeOrder, storage, resultCompute, right);
            }

            for(size_t i = 0; i < nFeatures; ++i)
            {
                NumericTablePtr parentHistogram = NumericTable::cast((*parentHistogramsForNode)[i]);
                ReadRows<algorithmFPType, cpu> parentHistogramRows(parentHistogram.get(), 0, binSizes[i]);
//                DAAL_CHECK_BLOCK_STATUS(parentHistogramRows);
                GHSumType *ghSumsParent = (GHSumType *)parentHistogramRows.get();
                GHSumType *ghSumsCompute = resultCompute->res[i].ghSums;
                GHSumType *ghSumsDiff = resultDiff->res[i].ghSums;

                hist::GHSumsHelper<algorithmFPType, int, BinIndexType, GHSumType, cpu>::computeDiff(binSizes[i], ghSumsParent, ghSumsCompute, ghSumsDiff);
            }

            for(size_t i = 0; i < nFeatures; ++i)
            {
                resultCompute->res[i].ghSums = nullptr;
                resultCompute->res[i].isReleased = true;
                resultDiff->res[i].ghSums = nullptr;
                resultDiff->res[i].isReleased = true;
            }

            resultCompute->res.~TVector<ResultType, cpu, ScalableAllocator<cpu>>();
            service_scalable_free<void, cpu>(resultCompute);
            resultDiff->res.~TVector<ResultType, cpu, ScalableAllocator<cpu>>();
            service_scalable_free<void, cpu>(resultDiff);
        }
        else
        {
            DataCollectionPtr histogramsForNode = DataCollection::cast((*dcHistograms)[histogramsOffsets[t]]);

            const TableRecord<algorithmFPType> *node = record.first ? record.first : record.second;

            MergedResult<ResultType, cpu> *result = new (services::internal::service_scalable_malloc<MergedResultType, cpu>(1)) MergedResultType(nFeatures);

            for(size_t i = 0; i < nFeatures; ++i)
            {
                services::Status s;
                services::SharedPtr<HomogenNumericTable<algorithmFPType> > ntHistogram = HomogenNumericTable<algorithmFPType>::create(4, binSizes[i], NumericTable::doAllocate, &s);
//                DAAL_CHECK_STATUS_VAR(s);
                (*histogramsForNode)[i] = ntHistogram;

                result->res[i].ghSums = (ghSum<algorithmFPType, cpu> *)ntHistogram->getArray();
            }

            computeHistograms<algorithmFPType, BinIndexType, cpu>(data, binSizes, optCoeffs, nRows, nFeatures, nDiffFeatMax, inputTreeOrder, storage, result, node);

            for(size_t i = 0; i < nFeatures; ++i)
            {
                result->res[i].ghSums = nullptr;
                result->res[i].isReleased = true;
            }
            result->res.~TVector<ResultType, cpu, ScalableAllocator<cpu>>();
            service_scalable_free<void, cpu>(result);
        }
    }});

    return services::Status();
}

//////////////////////////////////////////////////////////////////////////////////////////
// RegressionTrainDistrStep4Kernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, gbt::regression::training::Method method, CpuType cpu>
services::Status RegressionTrainDistrStep4Kernel<algorithmFPType, method, cpu>::packSplitIntoTable(const DAAL_INT idxFeatureBestSplit,
                                                                                                   const DAAL_INT featureIndex,
                                                                                                   const algorithmFPType impurityDecrease,
                                                                                                   const algorithmFPType leftGTotal,
                                                                                                   const algorithmFPType leftHTotal,
                                                                                                   const size_t          leftNTotal,
                                                                                                   const algorithmFPType rightGTotal,
                                                                                                   const algorithmFPType rightHTotal,
                                                                                                   const size_t          rightNTotal,
                                                                                                   NumericTablePtr &ntBestSplit)
{
    services::Status s;

    services::SharedPtr<HomogenNumericTable<algorithmFPType> > ntBestSplitHomogen = HomogenNumericTable<algorithmFPType>::create(1, 9, NumericTable::doAllocate, &s);
    DAAL_CHECK_STATUS_VAR(s);
    algorithmFPType * bestSplit = ntBestSplitHomogen->getArray();
    bestSplit[0] = (algorithmFPType) featureIndex;
    bestSplit[1] = (algorithmFPType) idxFeatureBestSplit;
    bestSplit[2] = impurityDecrease;
    bestSplit[3] = leftGTotal;
    bestSplit[4] = leftHTotal;
    bestSplit[5] = leftNTotal;
    bestSplit[6] = rightGTotal;
    bestSplit[7] = rightHTotal;
    bestSplit[8] = rightNTotal;

    ntBestSplit = ntBestSplitHomogen;

    return s;
}

template <typename algorithmFPType, gbt::regression::training::Method method, CpuType cpu>
services::Status RegressionTrainDistrStep4Kernel<algorithmFPType, method, cpu>::compute(      NumericTable   *ntInputTreeStructure,
                                                                                        const DataCollection *dcParentTotalHistogramsForFeatures,
                                                                                        const DataCollection *dcPartialHistogramsForFeatures,
                                                                                        const DataCollection *dcFeatureIndices,
                                                                                              DataCollection *dcTotalHistogramsForFeatures,
                                                                                              DataCollection *dcBestSplitsForFeatures,
                                                                                        const Parameter      &par)
{
    typedef ImpurityData<algorithmFPType, cpu> ImpurityType;
    typedef ghSum<algorithmFPType, cpu> GHSumType;
    typedef SplitData<algorithmFPType, ImpurityType> SplitDataType;
    typedef hist::Result<algorithmFPType, cpu> ResultType;
    typedef MergedResult<ResultType, cpu> MergedResultType;
    typedef int RowIndexType;
    typedef int BinIndexType;
    typedef hist::MaxImpurityDecreaseHelper<algorithmFPType, RowIndexType, BinIndexType, ImpurityType, GHSumType, SplitDataType, ResultType, cpu> MaxImpurityDecrease;
    typedef TreeTableConnector<algorithmFPType> ConnectorType;

    ConnectorType inputConnector(dynamic_cast<AOSNumericTable *>(ntInputTreeStructure));

    Collection<SplitRecord<algorithmFPType> > nodesForSplit;
    inputConnector.getSplitNodesMerged(0, nodesForSplit);
    const size_t nNodesForSplit = nodesForSplit.size();

    const size_t nFeatures = dcFeatureIndices->size();

    TArray<int, cpu> histogramsOffsets(nNodesForSplit);
    DAAL_CHECK_MALLOC(histogramsOffsets.get());

    size_t nHistograms = 0;

    for (size_t t = 0; t < nNodesForSplit; t++)
    {
        histogramsOffsets[t] = nHistograms;
        SplitRecord<algorithmFPType>& record = nodesForSplit[t];
        nHistograms += int(record.first != NULL) + int(record.second != NULL);
    }

    dcTotalHistogramsForFeatures->clear();
    dcBestSplitsForFeatures->clear();

    for (size_t t = 0; t < nFeatures; t++)
    {
        dcTotalHistogramsForFeatures->push_back(DataCollectionPtr(new DataCollection));
        dcBestSplitsForFeatures->push_back(DataCollectionPtr(new DataCollection));
    }

    LoopHelper<cpu>::run(true, nFeatures, [&](size_t id)
    {
        NumericTablePtr ntFeatureIndex = NumericTable::cast((*dcFeatureIndices)[id]);
        DataCollectionPtr dcParentTotalHistograms = DataCollection::cast((*dcParentTotalHistogramsForFeatures)[id]);
        DataCollectionPtr dcPartialHistograms = DataCollection::cast((*dcPartialHistogramsForFeatures)[id]);
        DataCollectionPtr dcTotalHistograms = DataCollection::cast((*dcTotalHistogramsForFeatures)[id]);
        DataCollectionPtr dcBestSplits = DataCollection::cast((*dcBestSplitsForFeatures)[id]);

        const size_t featureIndex = ntFeatureIndex->getValue<int>(0, 0);

        const size_t nCollections = dcPartialHistograms->size();
        
        dcTotalHistograms->clear();
        dcBestSplits->clear();

        for (size_t t = 0; t < nHistograms; t++)
        {
            dcTotalHistograms->push_back(NumericTablePtr());
            dcBestSplits->push_back(NumericTablePtr());
        }

        MergedResult<ResultType, cpu> *result = new (services::internal::service_scalable_malloc<MergedResultType, cpu>(1)) MergedResultType(nNodesForSplit);

        DAAL_ASSERT(DataCollection::cast((*dcPartialHistograms)[0])->size() == nHistograms);

        LoopHelper<cpu>::run(true, nNodesForSplit, [&](size_t t)
        {
            SplitRecord<algorithmFPType> &record = nodesForSplit[t];

            if (record.first == NULL && record.second == NULL)
            {
                return;
            }

            NumericTablePtr parentTotalHistogram = dcParentTotalHistograms ? NumericTable::cast((*dcParentTotalHistograms)[t]) : NumericTablePtr();

            const size_t curBins = NumericTable::cast((*DataCollection::cast((*dcPartialHistograms)[0]))[histogramsOffsets[t]])->getNumberOfRows();

            services::Status s;
            services::SharedPtr<HomogenNumericTable<algorithmFPType> > ntHistogram = HomogenNumericTable<algorithmFPType>::create(4, curBins, NumericTable::doAllocate, &s);
//            DAAL_CHECK_STATUS_VAR(s);
            algorithmFPType *histogram = ntHistogram->getArray();
        
            TArray<ReadRows<algorithmFPType, cpu>, cpu> partialHistogramRowsArray(nCollections);
//            DAAL_CHECK_MALLOC(partialHistogramRowsArray.get());
        
            TArray<algorithmFPType*, cpu> ptrs(nCollections);
//            DAAL_CHECK_MALLOC(ptrs.get());

            for (size_t i = 0; i < nCollections; i++)
            {
                NumericTablePtr ntPartialHistogram = NumericTable::cast((*DataCollection::cast((*dcPartialHistograms)[i]))[histogramsOffsets[t]]);
                partialHistogramRowsArray[i].set(ntPartialHistogram.get(), 0, curBins);
//                DAAL_CHECK_BLOCK_STATUS(partialHistogramRowsArray[i]);
                ptrs[i] = (algorithmFPType *)partialHistogramRowsArray[i].get();
            }

            hist::Result<algorithmFPType, cpu> &res = result->res[t];

            res.ghSums = (GHSumType *)histogram;
            res.gTotal = 0;
            res.hTotal = 0;
            res.iFeature = featureIndex;
            res.nUnique = curBins;

            hist::MergeGHSums<algorithmFPType, RowIndexType, BinIndexType, cpu>::run(curBins, 0, curBins, ptrs.get(), nCollections, res);

            if (record.first && record.second)
            {
                DAAL_ASSERT(parentTotalHistogram.get());

                const TableRecord<algorithmFPType> *left  = record.first;
                const TableRecord<algorithmFPType> *right = record.second;

                bool featureUnordered = false;
                SplitDataType splitLeft;
                SplitDataType splitRight;

                NumericTablePtr ntBestSplitLeft;
                NumericTablePtr ntBestSplitRight;

                DAAL_INT idxFeatureBestSplitLeft = -1;
                DAAL_INT idxFeatureBestSplitRight = -1;

                size_t nLeft = 0;

                for(size_t i = 0; i < curBins; ++i)
                {
                    nLeft += res.ghSums[i].n;
                }

                MaxImpurityDecrease::find(nLeft, par.minObservationsInLeafNode, par.lambda, splitLeft, res, idxFeatureBestSplitLeft, featureUnordered, featureIndex);

                /*DAAL_CHECK_STATUS_VAR(*/packSplitIntoTable(idxFeatureBestSplitLeft, featureIndex, splitLeft.impurityDecrease,
                                                         splitLeft.left.g, splitLeft.left.h, splitLeft.nLeft,
                                                         res.gTotal - splitLeft.left.g, res.hTotal - splitLeft.left.h, nLeft - splitLeft.nLeft,
                                                         ntBestSplitLeft);//);

                services::SharedPtr<HomogenNumericTable<algorithmFPType> > ntHistogramRight = HomogenNumericTable<algorithmFPType>::create(4, curBins, NumericTable::doAllocate, &s);
                algorithmFPType *histogramRight = ntHistogramRight->getArray();

                ReadRows<algorithmFPType, cpu> parentHistRows(parentTotalHistogram.get(), 0, curBins);
//                DAAL_CHECK_BLOCK_STATUS(parentHistRows);
                GHSumType *ghSumsParent = (GHSumType *)parentHistRows.get();
                GHSumType *ghSumsLeft = (GHSumType *)histogram;
                GHSumType *ghSumsRight = (GHSumType *)histogramRight;

                hist::GHSumsHelper<algorithmFPType, int, BinIndexType, GHSumType, cpu>::computeDiff(curBins, ghSumsParent, ghSumsLeft, ghSumsRight);

                res.ghSums = ghSumsRight;
                res.gTotal = 0;
                res.hTotal = 0;
                res.iFeature = featureIndex;
                res.nUnique = curBins;

                size_t nRight = 0;

                for(size_t i = 0; i < curBins; ++i)
                {
                    auto& sum = ghSumsRight[i];
                    res.gTotal += sum.g;
                    res.hTotal += sum.h;
                    nRight += sum.n;
                }

                MaxImpurityDecrease::find(nRight, par.minObservationsInLeafNode, par.lambda, splitRight, res, idxFeatureBestSplitRight, featureUnordered, featureIndex);

                /*DAAL_CHECK_STATUS_VAR(*/packSplitIntoTable(idxFeatureBestSplitRight, featureIndex, splitRight.impurityDecrease,
                                                         splitRight.left.g, splitRight.left.h, splitRight.nLeft,
                                                         res.gTotal - splitRight.left.g, res.hTotal - splitRight.left.h, nRight - splitRight.nLeft,
                                                         ntBestSplitRight);//);

                (*dcBestSplits)[histogramsOffsets[t] + 0] = ntBestSplitLeft;
                (*dcBestSplits)[histogramsOffsets[t] + 1] = ntBestSplitRight;

                (*dcTotalHistograms)[histogramsOffsets[t] + 0] = ntHistogram;
                (*dcTotalHistograms)[histogramsOffsets[t] + 1] = ntHistogramRight;
            }
            else
            {
                const TableRecord<algorithmFPType> *node = record.first ? record.first : record.second;

                bool featureUnordered = false;
                SplitDataType split;

                DAAL_INT idxFeatureBestSplit = -1;

                size_t n = 0;
                for(size_t i = 0; i < curBins; ++i)
                {
                    n += res.ghSums[i].n;
                }

                MaxImpurityDecrease::find(n, par.minObservationsInLeafNode, par.lambda, split, res, idxFeatureBestSplit, featureUnordered, featureIndex);

                NumericTablePtr ntBestSplit;
                /*DAAL_CHECK_STATUS_VAR(*/packSplitIntoTable(idxFeatureBestSplit, featureIndex, split.impurityDecrease,
                                                         split.left.g, split.left.h, split.nLeft, res.gTotal - split.left.g, res.hTotal - split.left.h, n - split.nLeft, ntBestSplit);//);

                (*dcBestSplits)[histogramsOffsets[t] + 0] = ntBestSplit;

                (*dcTotalHistograms)[histogramsOffsets[t] + 0] = ntHistogram;
            }

            res.ghSums = nullptr;
            res.isReleased = true;
        });

        result->res.~TVector<ResultType, cpu, ScalableAllocator<cpu>>();
        service_scalable_free<void, cpu>(result);
    });

    return services::Status();
}

//////////////////////////////////////////////////////////////////////////////////////////
// RegressionTrainDistrStep5Kernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, gbt::regression::training::Method method, CpuType cpu>
services::Status RegressionTrainDistrStep5Kernel<algorithmFPType, method, cpu>::unpackTableIntoSplit(DAAL_INT &idxFeatureBestSplit,
                                                                                                     DAAL_INT &featureIndex,
                                                                                                     algorithmFPType &impurityDecrease,
                                                                                                     algorithmFPType &leftGTotal,
                                                                                                     algorithmFPType &leftHTotal,
                                                                                                     size_t          &leftNTotal,
                                                                                                     algorithmFPType &rightGTotal,
                                                                                                     algorithmFPType &rightHTotal,
                                                                                                     size_t          &rightNTotal,
                                                                                                     const NumericTable *ntPartialBestSplit)
{
    ReadRows<algorithmFPType, cpu> partialBestSplitRows(const_cast<NumericTable *>(ntPartialBestSplit), 0, 9);
    DAAL_CHECK_BLOCK_STATUS(partialBestSplitRows);
    const algorithmFPType * const partialBestSplit = partialBestSplitRows.get();

    featureIndex        = (DAAL_INT)partialBestSplit[0];
    idxFeatureBestSplit = (DAAL_INT)partialBestSplit[1];
    impurityDecrease    = partialBestSplit[2];
    leftGTotal          = partialBestSplit[3];
    leftHTotal          = partialBestSplit[4];
    leftNTotal          = partialBestSplit[5];
    rightGTotal         = partialBestSplit[6];
    rightHTotal         = partialBestSplit[7];
    rightNTotal         = partialBestSplit[8];

    return services::Status();
}

template <typename algorithmFPType, gbt::regression::training::Method method, CpuType cpu>
services::Status RegressionTrainDistrStep5Kernel<algorithmFPType, method, cpu>::compute(const NumericTable   *ntBinnedData,
                                                                                        const NumericTable   *ntTransposedBinnedData,
                                                                                        const NumericTable   *ntBinSizes,
                                                                                              NumericTable   *ntInputTreeStructure,
                                                                                        const NumericTable   *ntInputTreeOrder,
                                                                                        const DataCollection *dcPartialBestSplits,
                                                                                              NumericTable   *ntTreeStructure,
                                                                                              NumericTable   *ntTreeOrder,
                                                                                        const Parameter      &par)
{
    const size_t nRows = ntBinnedData->getNumberOfRows();
    const size_t nFeatures = ntBinnedData->getNumberOfColumns();

    ReadRows<int, cpu> binSizesRows(const_cast<NumericTable *>(ntBinSizes), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(binSizesRows);
    const int * const binSizes = binSizesRows.get();

    int maxBinSize = 0;

    for (size_t i = 0; i < nFeatures; i++)
    {
        if (binSizes[i] > maxBinSize)
        {
            maxBinSize = binSizes[i];
        }
    }

    if (maxBinSize <= 256)
    {
        HomogenNumericTable<unsigned char> *ntBinnedData8 = dynamic_cast<HomogenNumericTable<unsigned char> *>(const_cast<NumericTable *>(ntBinnedData));
        HomogenNumericTable<unsigned char> *ntTransposedBinnedData8 = dynamic_cast<HomogenNumericTable<unsigned char> *>(const_cast<NumericTable *>(ntTransposedBinnedData));
        if (ntBinnedData8 && ntTransposedBinnedData8)
        {
            const uint8_t *data = ntBinnedData8->getArray();
            const uint8_t *transposedData = ntTransposedBinnedData8->getArray();
            return computeImpl<uint8_t>(data, transposedData, ntBinSizes, ntInputTreeStructure, ntInputTreeOrder, dcPartialBestSplits, ntTreeStructure, ntTreeOrder, par);
        }
    }

    if (maxBinSize <= 65536)
    {
        HomogenNumericTable<unsigned short> *ntBinnedData16 = dynamic_cast<HomogenNumericTable<unsigned short> *>(const_cast<NumericTable *>(ntBinnedData));
        HomogenNumericTable<unsigned short> *ntTransposedBinnedData16 = dynamic_cast<HomogenNumericTable<unsigned short> *>(const_cast<NumericTable *>(ntTransposedBinnedData));
        if (ntBinnedData16 && ntTransposedBinnedData16)
        {
            const uint16_t *data = ntBinnedData16->getArray();
            const uint16_t *transposedData = ntTransposedBinnedData16->getArray();
            return computeImpl<uint16_t>(data, transposedData, ntBinSizes, ntInputTreeStructure, ntInputTreeOrder, dcPartialBestSplits, ntTreeStructure, ntTreeOrder, par);

        }
    }

    ReadRows<int, cpu> ntBinnedDataRows(const_cast<NumericTable *>(const_cast<NumericTable *>(ntBinnedData)), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(ntBinnedDataRows);
    const int * const data = ntBinnedDataRows.get();

    ReadRows<int, cpu> ntTransposedBinnedDataRows(const_cast<NumericTable *>(const_cast<NumericTable *>(ntTransposedBinnedData)), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(ntTransposedBinnedDataRows);
    const int * const transposedData = ntTransposedBinnedDataRows.get();

    return computeImpl<int>(data, transposedData, ntBinSizes, ntInputTreeStructure, ntInputTreeOrder, dcPartialBestSplits, ntTreeStructure, ntTreeOrder, par);
}

template <typename algorithmFPType, gbt::regression::training::Method method, CpuType cpu>
template <typename BinIndexType>
services::Status RegressionTrainDistrStep5Kernel<algorithmFPType, method, cpu>::computeImpl(const BinIndexType   *binnedData,
                                                                                            const BinIndexType   *transposedBinnedData,
                                                                                            const NumericTable   *ntBinSizes,
                                                                                                  NumericTable   *ntInputTreeStructure,
                                                                                            const NumericTable   *ntInputTreeOrder,
                                                                                            const DataCollection *dcPartialBestSplits,
                                                                                                  NumericTable   *ntTreeStructure,
                                                                                                  NumericTable   *ntTreeOrder,
                                                                                            const Parameter      &par)
{
    typedef ImpurityData<algorithmFPType, cpu> ImpurityType;
    typedef ghSum<algorithmFPType, cpu> GHSumType;
    typedef SplitData<algorithmFPType, ImpurityType> SplitDataType;
    typedef hist::Result<algorithmFPType, cpu> ResultType;
    typedef MergedResult<ResultType, cpu> MergedResultType;
    typedef int RowIndexType;
    typedef typename TreeBuilder<algorithmFPType, RowIndexType, BinIndexType, cpu>::BestSplit BestSplitType;
    typedef hist::MaxImpurityDecreaseHelper<algorithmFPType, RowIndexType, BinIndexType, ImpurityType, GHSumType, SplitDataType, ResultType, cpu> MaxImpurityDecrease;
    typedef dtrees::internal::TVector<RowIndexType, cpu> IndexTypeArray;
    typedef DefaultPartitionTask<algorithmFPType, RowIndexType, BinIndexType, cpu> PartitionTaskType;
    typedef TreeTableConnector<algorithmFPType> ConnectorType;

    const size_t nRows = ntInputTreeOrder->getNumberOfRows();
    const size_t nFeatures = ntBinSizes->getNumberOfColumns();

    bool isTreeStructureInplace = (ntInputTreeStructure == ntTreeStructure);

    DAAL_ASSERT(isTreeStructureInplace);

    ConnectorType connector(dynamic_cast<AOSNumericTable *>(ntTreeStructure));

    if (!isTreeStructureInplace)
    {
        ConnectorType inputConnector(dynamic_cast<AOSNumericTable *>(ntInputTreeStructure));
        // TODO COPY input tree structure into output tree structure
    }

    Collection<TableRecord<algorithmFPType> *> nodesForSplit;
    connector.getSplitNodes(0, nodesForSplit);
    const size_t nNodesForSplit = nodesForSplit.size();

    const size_t nCollections = dcPartialBestSplits->size();

    IndexTypeArray bestSplitIdxBuf(nRows * 2);
    DAAL_CHECK_MALLOC(bestSplitIdxBuf.get());

    WriteRows<int, cpu> treeOrderRows(ntTreeOrder, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(treeOrderRows);
    int * const treeOrder = treeOrderRows.get();

    if (ntTreeOrder != ntInputTreeOrder)
    {
        ReadRows<int, cpu> inputTreeOrderRows(const_cast<NumericTable *>(ntInputTreeOrder), 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(inputTreeOrderRows);
        const int * const inputTreeOrder = inputTreeOrderRows.get();
        services::daal_memcpy_s(treeOrder, sizeof(int) * nRows, inputTreeOrder, sizeof(int) * nRows);
    }

//    for (size_t t = 0; t < nNodesForSplit; t++)
    LoopHelper<cpu>::run(true, nNodesForSplit, [&](size_t t)
    {
        DAAL_ASSERT(DataCollection::cast((*dcPartialBestSplits)[0])->size() == nNodesForSplit);
        TableRecord<algorithmFPType> *record = nodesForSplit[t];

        SplitDataType split;
        BestSplitType bestSplit(split, nullptr);

        DAAL_INT idxFeatureBestSplit;
        DAAL_INT featureIndex;
        RowIndexType nLeft;
        algorithmFPType leftGTotal;
        algorithmFPType leftHTotal;
        size_t          leftNTotal;
        algorithmFPType rightGTotal;
        algorithmFPType rightHTotal;
        size_t          rightNTotal;

        for (size_t i = 0; i < nCollections; i++)
        {
            NumericTablePtr ntPartialBestSplit = NumericTable::cast((*DataCollection::cast((*dcPartialBestSplits)[i]))[t]);
            SplitDataType partialBestSplit;
            algorithmFPType partialLeftGTotal;
            algorithmFPType partialLeftHTotal;
            size_t          partialLeftNTotal;
            algorithmFPType partialRightGTotal;
            algorithmFPType partialRightHTotal;
            size_t          partialRightNTotal;
            /*DAAL_CHECK_STATUS_VAR(*/unpackTableIntoSplit(idxFeatureBestSplit, featureIndex, partialBestSplit.impurityDecrease,
                                                       partialLeftGTotal, partialLeftHTotal, partialLeftNTotal,
                                                       partialRightGTotal, partialRightHTotal, partialRightNTotal,
                                                       ntPartialBestSplit.get());//);
            if (bestSplit.update(partialBestSplit, idxFeatureBestSplit, featureIndex))
            {
                leftGTotal  = partialLeftGTotal;
                leftHTotal  = partialLeftHTotal;
                leftNTotal  = partialLeftNTotal;
                rightGTotal = partialRightGTotal;
                rightHTotal = partialRightHTotal;
                rightNTotal = partialRightNTotal;
            }
        }

        bestSplit.getResult(featureIndex, idxFeatureBestSplit);

        if (featureIndex >= 0)
        {
            if (record->nid == 0)
            {
                record->gTotal = leftGTotal + rightGTotal;
                record->hTotal = leftHTotal + rightHTotal;
                record->nTotal = leftNTotal + rightNTotal;
            }

            algorithmFPType impDec;
            int iFeature;
            bestSplit.safeGetData(impDec, iFeature);

            const ImpurityType imp(record->gTotal, record->hTotal);

            impDec -= imp.value(par.lambda);
            if(impDec >= par.minSplitLoss)
            {
                PartitionTaskType::doPartitionIdxWithStride(record->n, treeOrder + record->iStart, transposedBinnedData + featureIndex * nRows, false /* featureUnordered */, idxFeatureBestSplit,
                                                  bestSplitIdxBuf.get() + 2 * record->iStart, nLeft, 1, 0);
                
                record->featureValue = idxFeatureBestSplit;
                record->featureIdx = featureIndex;
                record->featureUnordered = false;
                record->isFinalized = true;

                connector.createNode(record->level + 1, record->nid * 2 + 1, nLeft, record->iStart, leftGTotal, leftHTotal, leftNTotal, par);
                connector.createNode(record->level + 1, record->nid * 2 + 2, record->n - nLeft, record->iStart + nLeft, rightGTotal, rightHTotal, rightNTotal, par);
            }
            else
            {
                featureIndex = -1;
            }
        }

        if (featureIndex == -1)
        {
            record->isFinalized = true;
            record->nodeState = ConnectorType::badSplit;
        }
    });

    return services::Status();
}

//////////////////////////////////////////////////////////////////////////////////////////
// RegressionTrainDistrStep6Kernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, gbt::regression::training::Method method, CpuType cpu>
services::Status RegressionTrainDistrStep6Kernel<algorithmFPType, method, cpu>::compute(const NumericTable           *ntInitialResponse,
                                                                                        const DataCollection         *dcBinValues,
                                                                                        const DataCollection         *dcFinalizedTrees,
                                                                                              gbt::regression::Model *model,
                                                                                        const Parameter              &par)
{
    typedef TreeTableConnector<algorithmFPType> ConnectorType;

    const size_t nTrees = dcFinalizedTrees->size();
    const size_t nFeatures = dcBinValues->size();

    gbt::internal::ModelImpl &modelImpl = *static_cast<daal::algorithms::gbt::regression::internal::ModelImpl*>(model);

    modelImpl.reserve(nTrees);

    TArray<ReadRows<algorithmFPType, cpu>, cpu> binValuesRows(nFeatures);
    DAAL_CHECK_MALLOC(binValuesRows.get());
    TArray<algorithmFPType*, cpu> binValues(nFeatures);
    DAAL_CHECK_MALLOC(binValues.get());

    for (size_t i = 0; i < nFeatures; i++)
    {
        NumericTablePtr featureBinValues = NumericTable::cast((*dcBinValues)[i]);
        binValuesRows[i].set(featureBinValues.get(), 0, 1);
        DAAL_CHECK_BLOCK_STATUS(binValuesRows[i]);
        binValues[i] = (algorithmFPType *)binValuesRows[i].get();
    }

    algorithmFPType initialF = 0.0;

    if (ntInitialResponse)
    {
        initialF = ntInitialResponse->getValue<algorithmFPType>(0, 0);
    }

    for (size_t i = 0; i < nTrees; i++)
    {
        NumericTablePtr ntTreeStructure = NumericTable::cast((*dcFinalizedTrees)[i]);
        ConnectorType connector(dynamic_cast<AOSNumericTable *>(ntTreeStructure.get()));

        size_t maxLevel = 0;
        connector.getMaxLevel(0, maxLevel);
        const size_t nNodes = (1 << (maxLevel + 1)) - 1;
        const size_t nNodesPresent = connector.getNNodes(0);

        gbt::internal::GbtDecisionTree *pTbl = new gbt::internal::GbtDecisionTree(nNodes, maxLevel, nNodesPresent);

        HomogenNumericTable<double> *pTblImp     = new HomogenNumericTable<double>(1, nNodes, NumericTable::doAllocate);
        HomogenNumericTable<int>    *pTblSmplCnt = new HomogenNumericTable<int>(1, nNodes, NumericTable::doAllocate);

        connector.convertToGbtDecisionTree<cpu>(binValues.get(), nNodes, maxLevel, pTbl, pTblImp->getArray(), pTblSmplCnt->getArray(), initialF, par);
        modelImpl.add(pTbl, pTblImp, pTblSmplCnt);

        initialF = 0.0;
    }

    return services::Status();
}

} /* namespace internal */
} /* namespace training */
} /* namespace regression */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif
