/* file: zscore_impl.i */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

//++
//  Implementation of template function that calculates zscore normalization.
//--

#ifndef __ZSCORE_IMPL_I__
#define __ZSCORE_IMPL_I__

#include "zscore_base.h"
#include "service_math.h"
#include "service_memory.h"
#include "service_numeric_table.h"
#include "threading.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::services::internal;

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
void ZScoreKernelBase<algorithmFPType, cpu>::compute(SharedPtr<NumericTable> inputTable,  NumericTable *sumTable, NumericTable *resultTable,
                                                     daal::algorithms::Parameter *parameter)
{
    size_t _nVectors    = inputTable->getNumberOfRows();
    size_t _nFeatures   = inputTable->getNumberOfColumns();

#define _BLOCK_SIZE_NORM_ 256

    /* Split rows by blocks, block size cannot be less than _nVectors */
    size_t numRowsInBlock = (_nVectors > _BLOCK_SIZE_NORM_)?_BLOCK_SIZE_NORM_:_nVectors;
    /* Number of blocks */
    size_t numRowsBlocks   = _nVectors / numRowsInBlock;
    /* Last block can be bigger than others */
    size_t numRowsInLastBlock = numRowsInBlock + ( _nVectors - numRowsBlocks * numRowsInBlock);

    /* Internal arrays for mean and variance, initialized by zeros */
    algorithmFPType* mean_total      = service_calloc<algorithmFPType, cpu>(_nFeatures);
    algorithmFPType* inv_sigma_total = service_calloc<algorithmFPType, cpu>(_nFeatures);
    if(!(mean_total) || !(inv_sigma_total))
    {
        this->_errors->add(daal::services::ErrorMemoryAllocationFailed);

        if(mean_total)      service_free<algorithmFPType,cpu>( mean_total );
        if(inv_sigma_total) service_free<algorithmFPType,cpu>( inv_sigma_total );

        return;
    }

    /* Check if input data are already normalized */
    if (inputTable->isNormalized(NumericTableIface::standardScoreNormalized))
    {
        /* In case of non-inplace just copy input array to output */
        if(inputTable.get() != resultTable)
        {
            daal::threader_for( numRowsBlocks, numRowsBlocks, [ & ](int iRowsBlock)
            {
                size_t _nRows    = (iRowsBlock < (numRowsBlocks-1))?numRowsInBlock:numRowsInLastBlock;
                size_t _startRow = iRowsBlock * numRowsInBlock;

                daal::internal::ReadRows<algorithmFPType, cpu, NumericTable> dataTableBD( inputTable.get(), _startRow, _nRows );
                const algorithmFPType* dataArray_local = dataTableBD.get();

                daal::internal::WriteOnlyRows<algorithmFPType, cpu, NumericTable> normDataTableBD( resultTable, _startRow, _nRows );
                algorithmFPType* normDataArray_local = normDataTableBD.get();

                for(int i = 0; i < _nRows; i++)
                {
                   PRAGMA_IVDEP
                   PRAGMA_VECTOR_ALWAYS
                    for(int j = 0; j < _nFeatures; j++)
                    {
                        normDataArray_local[i * _nFeatures + j] = dataArray_local[i * _nFeatures + j];
                    }
                }
            } );

            resultTable->setNormalizationFlag(NumericTableIface::standardScoreNormalized);
        }

        return;
    }

    /* Call method-specific function to compute means and variances */
    if( computeMeanVariance_thr( inputTable, mean_total, inv_sigma_total, parameter ) )
    {
        if(mean_total)      service_free<algorithmFPType,cpu>( mean_total );
        if(inv_sigma_total) service_free<algorithmFPType,cpu>( inv_sigma_total );

        return;
    }

    /* Final normalization threaded loop */
    daal::threader_for( numRowsBlocks, numRowsBlocks, [ & ](int iRowsBlock)
    {
        size_t _nRows    = (iRowsBlock < (numRowsBlocks-1))?numRowsInBlock:numRowsInLastBlock;
        size_t _startRow = iRowsBlock * numRowsInBlock;

        daal::internal::ReadRows<algorithmFPType, cpu, NumericTable> dataTableBD( inputTable.get(), _startRow, _nRows );
        const algorithmFPType* dataArray_local = dataTableBD.get();

        daal::internal::WriteOnlyRows<algorithmFPType, cpu, NumericTable> normDataTableBD( resultTable, _startRow, _nRows );
        algorithmFPType* normDataArray_local = normDataTableBD.get();

        for(int i = 0; i < _nRows; i++)
        {
           PRAGMA_IVDEP
           PRAGMA_VECTOR_ALWAYS
            for(int j = 0; j < _nFeatures; j++)
            {
                normDataArray_local[i * _nFeatures + j] = (dataArray_local[i * _nFeatures + j] - mean_total[j]) * inv_sigma_total[j];
            }
        }
    } );

    service_free<algorithmFPType,cpu>( mean_total );
    service_free<algorithmFPType,cpu>( inv_sigma_total );

    resultTable->setNormalizationFlag(NumericTableIface::standardScoreNormalized);

    return;
};

} // namespace daal::internal
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
