/* file: sorting_impl.i */
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

template<Method method, typename algorithmFPType, CpuType cpu>
Status SortingKernel<method, algorithmFPType, cpu>::compute(const NumericTable &inputTable, NumericTable &outputTable)
{
    const size_t nFeatures = inputTable.getNumberOfColumns();
    const size_t nVectors  = inputTable.getNumberOfRows();

    ReadRows<algorithmFPType, cpu> inputBlock(const_cast<NumericTable &>(inputTable), 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(inputBlock);
    const algorithmFPType *data = inputBlock.get();

    WriteOnlyRows<algorithmFPType, cpu> otputBlock(outputTable, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(otputBlock);
    algorithmFPType *sortedData = otputBlock.get();

    DAAL_CHECK(!(Statistics<algorithmFPType, cpu>::xSort(const_cast<algorithmFPType *>(data), nFeatures, nVectors, sortedData)), ErrorSorting);
    return Status();
}

} // namespace daal::algorithms::sorting::internal
} // namespace daal::algorithms::sorting
} // namespace daal::algorithms
} // namespace daal

#endif
