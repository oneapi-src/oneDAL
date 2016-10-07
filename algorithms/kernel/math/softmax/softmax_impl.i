/* file: softmax_impl.i */
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

/*
//++
//  Implementation of softmax algorithm
//--
*/

#include "service_data_utils.h"
#include "service_math.h"
#include "threading.h"

using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace math
{
namespace softmax
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
inline void SoftmaxKernel<algorithmFPType, method, cpu>::processBlock(NumericTable *inputTable, size_t nInputColumns,
                                                                      size_t nProcessedRows, size_t nRowsInCurrentBlock,
                                                                      NumericTable *resultTable)
{
    BlockDescriptor<algorithmFPType> inputBlock;
    inputTable->getBlockOfRows(nProcessedRows, nRowsInCurrentBlock, readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getBlockPtr();

    BlockDescriptor<algorithmFPType> resultBlock;
    resultTable->getBlockOfRows(nProcessedRows, nRowsInCurrentBlock, writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getBlockPtr();

    algorithmFPType minValue = -data_feature_utils::internal::MaxVal<algorithmFPType, cpu>::get();
    algorithmFPType max;
    algorithmFPType sum = (algorithmFPType)0;

    for(size_t i = 0; i < nRowsInCurrentBlock; i++)
    {
        max = minValue;

        for(size_t j = 0; j < nInputColumns; j++)
        {
            if(max < inputArray[i * nInputColumns + j])
            {
                max = inputArray[i * nInputColumns + j];
            }
            resultArray[i * nInputColumns + j] = inputArray[i * nInputColumns + j];
        }

        for(size_t j = 0; j < nInputColumns; j++)
        {
            resultArray[i * nInputColumns + j] -= max;
        }

        daal::internal::Math<algorithmFPType,cpu>::vExp(nInputColumns, resultArray + i * nInputColumns, resultArray + i * nInputColumns);

        for(size_t j = 0; j < nInputColumns; j++)
        {
            sum += resultArray[i * nInputColumns + j];
        }

        for(size_t j = 0; j < nInputColumns; j++)
        {
            resultArray[i * nInputColumns + j] /= sum;
        }
        sum = (algorithmFPType)0;
    }

    inputTable->releaseBlockOfRows(inputBlock);
    resultTable->releaseBlockOfRows(resultBlock);
}

/**
 *  \brief Kernel for Softmax calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void SoftmaxKernel<algorithmFPType, method, cpu>::compute(NumericTable *inputTable, NumericTable *resultTable)
{
    size_t nInputRows    = inputTable->getNumberOfRows();
    size_t nInputColumns = inputTable->getNumberOfColumns();

    size_t nBlocks = nInputRows / _nRowsInBlock;
    nBlocks += (nBlocks * _nRowsInBlock != nInputRows);

    daal::threader_for(nBlocks, nBlocks, [ = ](int block)
    {
        size_t nRowsToProcess = _nRowsInBlock;
        if( block == nBlocks - 1 )
        {
            nRowsToProcess = nInputRows - block * _nRowsInBlock;
        }

        processBlock(inputTable, nInputColumns, block * _nRowsInBlock, nRowsToProcess, resultTable);
    } );
}

} // namespace daal::internal
} // namespace softmax
} // namespace math
} // namespace algorithms
} // namespace daal
