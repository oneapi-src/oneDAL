/* file: zscore_dense_sum_impl.i */
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
//  Implementation of sumDense method for zscore normalization algorithm
//--
*/

#ifndef __ZSCORE_DENSE_SUM_IMPL_I__
#define __ZSCORE_DENSE_SUM_IMPL_I__

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{
namespace internal
{


template<typename algorithmFPType, CpuType cpu>
Status ZScoreKernel<algorithmFPType, sumDense, cpu>::computeMeanVariance_thr(NumericTable& inputTable,
                                                                             algorithmFPType* resultMean,
                                                                             algorithmFPType* resultVariance,
                                                                             const daal::algorithms::Parameter &parameter)
{
    const size_t nVectors = inputTable.getNumberOfRows();
    const size_t nFeatures = inputTable.getNumberOfColumns();

    NumericTablePtr sumTable = inputTable.basicStatistics.get(NumericTableIface::sum);
    DAAL_CHECK(sumTable, ErrorPrecomputedSumNotAvailable);

    ReadRows<algorithmFPType, cpu, NumericTable> sumBlock(sumTable.get(), 0, 1);
    const algorithmFPType* sumArray = sumBlock.get();

    algorithmFPType invN = algorithmFPType(1.0) / algorithmFPType(nVectors);
    algorithmFPType invNm1 = algorithmFPType(1.0) / (algorithmFPType(nVectors) - algorithmFPType(1.0));

    /* Compute means from sums */
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (int j = 0; j < nFeatures; j++)
    {
        resultMean[j] = sumArray[j] * invN;
        resultVariance[j] = 0;
    }

#define _BLOCK_SIZE_ 256

    /* Split rows by blocks, block size cannot be less than nVectors */
    size_t numRowsInBlock = (nVectors > _BLOCK_SIZE_)?_BLOCK_SIZE_:nVectors;
    /* Number of blocks */
    size_t numBlocks   = nVectors / numRowsInBlock;
    /* Last block can be bigger than others */
    size_t numRowsInLastBlock = numRowsInBlock + (nVectors - numBlocks * numRowsInBlock);

    /* TLS data initialization */
    daal::tls<algorithmFPType *> tls_data([&]()
    {
        return service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
    });

    SafeStatus safeStat;
    /* Compute partial unscaled variances for each block */
    daal::threader_for( numBlocks, numBlocks, [&](int iBlock)
    {
        algorithmFPType* pVariances = tls_data.local();
        DAAL_CHECK_THR(pVariances, ErrorMemoryAllocationFailed);

        size_t _nRows    = (iBlock < (numBlocks-1))?numRowsInBlock:numRowsInLastBlock;
        size_t _startRow = iBlock * numRowsInBlock;

        ReadRows<algorithmFPType, cpu, NumericTable> dataTableBD(inputTable, _startRow, _nRows);
        DAAL_CHECK_BLOCK_STATUS_THR(dataTableBD);
        const algorithmFPType* dataArray_local = dataTableBD.get();

        for(int i = 0; i < _nRows; i++)
        {
           PRAGMA_IVDEP
           PRAGMA_VECTOR_ALWAYS
            for(int j = 0; j < nFeatures; j++)
            {
                algorithmFPType _v = dataArray_local[i*nFeatures + j] - resultMean[j];
                pVariances[j]  +=  (_v * _v);
            }
        }
    } );

    /* Merge unscaled variance arrays by blocks */
    tls_data.reduce( [&]( algorithmFPType* pVariances)
    {
        if (pVariances)
        {
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (int j = 0; j < nFeatures; j++)
            {
                resultVariance[j] += pVariances[j];
            }
        }
        service_scalable_free<algorithmFPType, cpu>(pVariances);
    } );
    /* Convert array of variances to unbiased */

    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for(int j = 0; j < nFeatures; j++)
    {
        resultVariance[j] *= invNm1;
    }

    return safeStat.detach();

}

} // namespace internal
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
