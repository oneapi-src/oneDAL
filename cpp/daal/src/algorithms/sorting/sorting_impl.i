/* file: sorting_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Sorting observations algorithm implementation
//--
*/

#ifndef __SORTING_IMPL__
#define __SORTING_IMPL__

namespace daal
{
namespace algorithms
{
namespace sorting
{
namespace internal
{
template <Method method, typename algorithmFPType, CpuType cpu>
Status SortingKernel<method, algorithmFPType, cpu>::compute(const NumericTable & inputTable, NumericTable & outputTable)
{
    const size_t nFeatures = inputTable.getNumberOfColumns();
    const size_t nVectors  = inputTable.getNumberOfRows();

    ReadRows<algorithmFPType, cpu> inputBlock(const_cast<NumericTable &>(inputTable), 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(inputBlock);
    const algorithmFPType * data = inputBlock.get();

    WriteOnlyRows<algorithmFPType, cpu> otputBlock(outputTable, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(otputBlock);
    algorithmFPType * sortedData = otputBlock.get();

    DAAL_CHECK(!(StatisticsInst<algorithmFPType, cpu>::xSort(const_cast<algorithmFPType *>(data), nFeatures, nVectors, sortedData)), ErrorSorting);
    return Status();
}

} // namespace internal
} // namespace sorting
} // namespace algorithms
} // namespace daal

#endif
