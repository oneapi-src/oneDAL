/* file: tanh_dense_default_impl.i */
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
//  Implementation of hyperbolic tangent algorithm
//--
*/

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
inline Status TanhKernel<algorithmFPType, defaultDense, cpu>::processBlock(const NumericTable &inputTable, size_t nInputColumns,
                                                                           size_t nProcessedRows, size_t nRowsInCurrentBlock,
                                                                           NumericTable &resultTable)
{
    ReadRows<algorithmFPType, cpu, NumericTable> inputBlock(const_cast<NumericTable *>(&inputTable), nProcessedRows, nRowsInCurrentBlock);
    DAAL_CHECK_BLOCK_STATUS(inputBlock);
    const algorithmFPType* inputArray = inputBlock.get();

    WriteRows<algorithmFPType, cpu, NumericTable> resultBlock(&resultTable, nProcessedRows, nRowsInCurrentBlock);
    DAAL_CHECK_BLOCK_STATUS(resultBlock);
    algorithmFPType* resultArray = resultBlock.get();

    size_t nDataElements = nInputColumns * nRowsInCurrentBlock;
    daal::internal::Math<algorithmFPType,cpu>::vTanh(nDataElements, const_cast<algorithmFPType *>(inputArray), resultArray);
    return Status();
}

} // namespace daal::internal
} // namespace tanh
} // namespace math
} // namespace algorithms
} // namespace daal
