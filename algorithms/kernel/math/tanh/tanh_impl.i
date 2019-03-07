/* file: tanh_impl.i */
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

/**
 *  \brief Kernel for Hyperbolic Tangent calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
services::Status TanhKernelBase<algorithmFPType, method, cpu>::compute(const NumericTable *inputTable, NumericTable *resultTable)
{
    size_t nInputRows    = inputTable->getNumberOfRows();
    size_t nInputColumns = inputTable->getNumberOfColumns();

    size_t nBlocks = nInputRows / _nRowsInBlock;
    nBlocks += (nBlocks * _nRowsInBlock != nInputRows);

    SafeStatus safeStat;
    daal::threader_for(nBlocks, nBlocks, [ =, &safeStat ](int block)
    {
        size_t nRowsToProcess = _nRowsInBlock;
        if( block == nBlocks - 1 )
        {
            nRowsToProcess = nInputRows - block * _nRowsInBlock;
        }

        safeStat |= processBlock(*inputTable, nInputColumns, block * _nRowsInBlock, nRowsToProcess, *resultTable);
    } );
    return safeStat.detach();
}

} // namespace daal::internal
} // namespace tanh
} // namespace math
} // namespace algorithms
} // namespace daal
