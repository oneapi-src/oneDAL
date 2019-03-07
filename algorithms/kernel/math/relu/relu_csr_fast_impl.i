/* file: relu_csr_fast_impl.i */
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
//  Implementation of relu algorithm
//--
*/

namespace daal
{
namespace algorithms
{
namespace math
{
namespace relu
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
inline Status ReLUKernel<algorithmFPType, fastCSR, cpu>::processBlock(const NumericTable &inputTable, size_t nInputColumns,
                                                                      size_t nProcessedRows, size_t nRowsInCurrentBlock,
                                                                      NumericTable &resultTable)
{
    CSRNumericTableIface *inTable  = dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(&inputTable));
    CSRNumericTableIface *resTable = dynamic_cast<CSRNumericTableIface *>(&resultTable);

    ReadRowsCSR<algorithmFPType, cpu> inputBlock(inTable, nProcessedRows, nRowsInCurrentBlock);
    DAAL_CHECK_BLOCK_STATUS(inputBlock);
    const algorithmFPType* inputArray = inputBlock.values();

    WriteRowsCSR<algorithmFPType, cpu> resultBlock(resTable, nProcessedRows, nRowsInCurrentBlock);
    DAAL_CHECK_BLOCK_STATUS(resultBlock);
    algorithmFPType* resultArray = resultBlock.values();

    const size_t nDataElements = resultBlock.size();
    for(size_t i = 0; i < nDataElements; i++)
    {
        if(inputArray[i] >= (algorithmFPType)0)
        {
            resultArray[i] = inputArray[i];
        }
        else
        {
            resultArray[i] = (algorithmFPType)0;
        }
    }
    return Status();
}

} // namespace daal::internal
} // namespace relu
} // namespace math
} // namespace algorithms
} // namespace daal
