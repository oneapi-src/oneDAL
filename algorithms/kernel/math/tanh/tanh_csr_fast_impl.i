/* file: tanh_csr_fast_impl.i */
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
//  Implementation of hyperbolic tangent algorithm
//--
*/

#include "service_math.h"

using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace math
{
namespace tanh
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
inline void TanhKernel<algorithmFPType, fastCSR, cpu>::processBlock(NumericTable* inputTable, size_t nInputColumns,
                                                                    size_t nProcessedRows, size_t nRowsInCurrentBlock,
                                                                    NumericTable* resultTable)
{
    CSRNumericTableIface* inTable = dynamic_cast<CSRNumericTableIface*>(inputTable);
    CSRNumericTableIface* resTable = dynamic_cast<CSRNumericTableIface*>(resultTable);

    CSRBlockDescriptor<algorithmFPType> inputBlock;
    inTable->getSparseBlock(nProcessedRows, nRowsInCurrentBlock, readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getBlockValuesPtr();

    CSRBlockDescriptor<algorithmFPType> resultBlock;
    resTable->getSparseBlock(nProcessedRows, nRowsInCurrentBlock, writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getBlockValuesPtr();

    size_t nDataElements = resultBlock.getDataSize();
    daal::internal::Math<algorithmFPType,cpu>::vTanh(nDataElements, inputArray, resultArray);

    inTable->releaseSparseBlock(inputBlock);
    resTable->releaseSparseBlock(resultBlock);
}

} // namespace daal::internal
} // namespace tanh
} // namespace math
} // namespace algorithms
} // namespace daal
