/* file: linear_model_train_normeq_merge_impl.i */
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
//  Implementation of common base classes for normal equations model training.
//--
*/

#include "linear_model_train_normeq_kernel.h"
#include "threading.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace normal_equations
{
namespace training
{
namespace internal
{
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services::internal;

template<CpuType cpu, typename F>
void conditional_threader_for(bool condition, size_t n, size_t threadsRequest, const F &processIteration)
{
    if (condition)
    {
        daal::threader_for(n, threadsRequest, processIteration);
    }
    else
    {
        for(size_t i = 0; i < n; i++ )
        {
            processIteration(i);
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
Status MergeKernel<algorithmFPType, cpu>::merge(const NumericTable &partialTable,
                                                algorithmFPType* result,
                                                bool threadingCondition)
{
    const size_t nRows = partialTable.getNumberOfRows();

    ReadRowsType block(const_cast<NumericTable &>(partialTable), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(block);
    algorithmFPType *partialResult = const_cast<algorithmFPType *>(block.get());

    size_t resultSize = nRows * partialTable.getNumberOfColumns();
    conditional_threader_for<cpu>(threadingCondition, resultSize, resultSize, [ = ](size_t i)
    {
        result[i] += partialResult[i];
    } );
    return Status();
}

template <typename algorithmFPType, CpuType cpu>
Status MergeKernel<algorithmFPType, cpu>::compute(size_t n, NumericTable **partialxtx,
                                                  NumericTable **partialxty,
                                                  NumericTable &xtxTable,
                                                  NumericTable &xtyTable)
{
    const size_t nBetas     = xtxTable.getNumberOfRows();
    const size_t nResponses = xtyTable.getNumberOfRows();

    WriteOnlyRowsType xtxBlock(xtxTable, 0, nBetas);
    DAAL_CHECK_BLOCK_STATUS(xtxBlock);
    algorithmFPType *xtx = xtxBlock.get();

    WriteOnlyRowsType xtyBlock(xtyTable, 0, nResponses);
    DAAL_CHECK_BLOCK_STATUS(xtyBlock);
    algorithmFPType *xty = xtyBlock.get();

    service_memset<algorithmFPType, cpu>(xtx, 0, nBetas * nBetas);
    service_memset<algorithmFPType, cpu>(xty, 0, nBetas * nResponses);

    const size_t minThreadingSize = 512 * 1024;
    Status st;
    for (size_t i = 0; i < n; i++)
    {
        st |= MergeKernel<algorithmFPType, cpu>::merge(*partialxtx[i], xtx,
                    nBetas * nBetas * sizeof(algorithmFPType) > minThreadingSize);
        DAAL_CHECK_STATUS_VAR(st);
        st |= MergeKernel<algorithmFPType, cpu>::merge(*partialxty[i], xty,
                    nBetas * nResponses * sizeof(algorithmFPType) > minThreadingSize);
        DAAL_CHECK_STATUS_VAR(st);
    }
    return st;
}

}
}
}
}
}
}
