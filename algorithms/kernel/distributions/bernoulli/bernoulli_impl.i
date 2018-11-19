/* file: bernoulli_impl.i */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
//  Implementation of bernoulli algorithm
//--
*/

#ifndef __BERNOULLI_IMPL_I__
#define __BERNOULLI_IMPL_I__

using namespace daal::algorithms::distributions::uniform::internal;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace distributions
{
namespace bernoulli
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
Status BernoulliKernel<algorithmFPType, method, cpu>::computeInt(int *resultArray, size_t n, algorithmFPType p, engines::BatchBase &engine)
{
    const int one  = 1;
    const int zero = 0;
    const size_t nElemsInBlock = 1024;

    size_t nBlocks = n / nElemsInBlock;
    nBlocks += (nBlocks * nElemsInBlock != n);

    algorithmFPType buffer[nElemsInBlock];

    size_t nElemsToProcess = nElemsInBlock;
    Status s;
    for(size_t block = 0; block < nBlocks; block++)
    {
        size_t nProcessed = block * nElemsInBlock;
        int *array = resultArray + nProcessed;
        if(block == nBlocks - 1)
        {
            nElemsToProcess = n - nProcessed;
        }
        DAAL_CHECK_STATUS(s, (UniformKernelDefault<algorithmFPType, cpu>::compute(0.0, 1.0, engine, nElemsToProcess, buffer)));

        for(size_t j = 0; j < nElemsToProcess; j++)
        {
            array[j] = ((buffer[j] < p) ? one : zero);
        }
    }
    return s;
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status BernoulliKernel<algorithmFPType, method, cpu>::computeFPType(NumericTable *resultTable, algorithmFPType p, engines::BatchBase &engine)
{
    const algorithmFPType one  = 1;
    const algorithmFPType zero = 0;
    const size_t nElemsInBlock = 1024;

    size_t nRows = resultTable->getNumberOfRows();
    size_t nCols = resultTable->getNumberOfColumns();

    Status s;

    if(nElemsInBlock > nCols)
    {
        size_t nRowsInBlock = nElemsInBlock / nCols;
        size_t nRowBlocks = nRows / nRowsInBlock;
        nRowBlocks += (nRowBlocks * nRowsInBlock != nRows);

        for(size_t block = 0; block < nRowBlocks; block++) /* blocking by rows */
        {
            size_t nProcessedRows = block * nRowsInBlock;
            size_t nRowsToProcess = (block == nRowBlocks - 1) ? (nRows - nProcessedRows) : nRowsInBlock;

            daal::internal::WriteRows<algorithmFPType, cpu> resultBlock(resultTable, nProcessedRows, nRowsToProcess);
            DAAL_CHECK_BLOCK_STATUS(resultBlock);
            algorithmFPType *resultArray = resultBlock.get();

            size_t nElemsToProcess = nRowsToProcess * nCols;
            DAAL_CHECK_STATUS(s, (UniformKernelDefault<algorithmFPType, cpu>::compute(0.0, 1.0, engine, nElemsToProcess, resultArray)));

            for(size_t j = 0; j < nElemsToProcess; j++)
            {
                resultArray[j] = ((resultArray[j] < p) ? one : zero);
            }
        }
    }
    else
    {
        size_t nColBlocks = nCols / nElemsInBlock;
        nColBlocks += (nColBlocks * nElemsInBlock != nCols);

        for(size_t block = 0; block < nRows; block++) /* blocking by rows */
        {
            daal::internal::WriteRows<algorithmFPType, cpu> resultBlock(resultTable, block, 1);
            DAAL_CHECK_BLOCK_STATUS(resultBlock);
            algorithmFPType *resultArray = resultBlock.get();

            for(size_t colBlock = 0; colBlock < nColBlocks; colBlock++) /* blocking within one column */
            {
                size_t nProcessedElems = nElemsInBlock * colBlock;
                size_t nElemsToProcess = (colBlock == nColBlocks - 1) ? (nCols - nProcessedElems) : nElemsInBlock;

                algorithmFPType *array = resultArray + nProcessedElems;
                DAAL_CHECK_STATUS(s, (UniformKernelDefault<algorithmFPType, cpu>::compute(0.0, 1.0, engine, nElemsToProcess, array)));

                for(size_t j = 0; j < nElemsToProcess; j++)
                {
                    array[j] = ((array[j] < p) ? one : zero);
                }
            }
        }
    }
    return s;
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status BernoulliKernel<algorithmFPType, method, cpu>::compute(algorithmFPType p, engines::BatchBase &engine, NumericTable *resultTable)
{
    NumericTableDictionary *resultDict = resultTable->getDictionary();
    if(resultDict->getFeaturesEqual() == DictionaryIface::equal && resultTable->getDataLayout() == NumericTableIface::aos &&
      (*resultDict)[0].indexType      == features::DAAL_INT32_S)  /* if table is HomogenNT<int> */
    {
        size_t n = resultTable->getNumberOfRows() * resultTable->getNumberOfColumns();
        int *resultArray = ((HomogenNumericTable<int> *)resultTable)->getArray();
        return computeInt(resultArray, n, p, engine);
    }
    else
    {
        return computeFPType(resultTable, p, engine);
    }
    return Status();
}

} // namespace internal
} // namespace bernoulli
} // namespace distributions
} // namespace algorithms
} // namespace daal

#endif
