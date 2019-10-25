/* file: gbt_regression_loss_impl.h */
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
//  Implementation of the class defining the gradient boosted trees loss function
//--
*/

#ifndef __GBT_REGRESSION_LOSS_IMPL__
#define __GBT_REGRESSION_LOSS_IMPL__

#include "gbt_train_aux.i"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace internal
{

using namespace daal::algorithms::gbt::training::internal;

//////////////////////////////////////////////////////////////////////////////////////////
// Squared loss function, L(y,f)=1/2([y-f(x)]^2)
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class SquaredLoss : public LossFunction<algorithmFPType, cpu>
{
public:
    virtual void getGradients(size_t n, size_t nRows, const algorithmFPType* y, const algorithmFPType* f,
        const IndexType* sampleInd, algorithmFPType* gh) DAAL_C11_OVERRIDE
    {
        const size_t nThreads  = daal::threader_get_threads_number();
        const size_t nBlocks   = getNBlocksForOpt<cpu>(nThreads, n);
        const size_t nPerBlock = n / nBlocks;
        const size_t nSurplus  = n % nBlocks;
        const bool inParallel  = nBlocks > 1;
        LoopHelper<cpu>::run(inParallel, nBlocks, [&](size_t iBlock) {
            const size_t start = iBlock + 1 > nSurplus ? nPerBlock * iBlock + nSurplus : (nPerBlock + 1) * iBlock;
            const size_t end   = iBlock + 1 > nSurplus ? start + nPerBlock : start + (nPerBlock + 1);
            if (sampleInd)
            {
                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = start; i < end; i++)
                {
                    gh[2 * sampleInd[i]]     = f[sampleInd[i]] - y[sampleInd[i]]; //gradient
                    gh[2 * sampleInd[i] + 1] = 1;                                 //hessian
                }
            }
            else
            {
                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = start; i < end; i++)
                {
                    gh[2 * i]     = f[i] - y[i]; //gradient
                    gh[2 * i + 1] = 1;           //hessian
                }
            }
        });
    }
};

} // namespace internal
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif
