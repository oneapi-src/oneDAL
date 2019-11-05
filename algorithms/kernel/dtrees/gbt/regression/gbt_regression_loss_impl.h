/* file: gbt_regression_loss_impl.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
