/* file: brownboost_predict_impl_v1.i */
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
//  Implementation of Fast method for Brown Boost prediction algorithm.
//--
*/

#ifndef __BROWNBOOST_PREDICT_IMPL_V1_I__
#define __BROWNBOOST_PREDICT_IMPL_V1_I__

#include "service_math.h"
#include "service_numeric_table.h"
#include "service_memory.h"

namespace daal
{
namespace algorithms
{
namespace brownboost
{
namespace prediction
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services::internal;

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status I1BrownBoostPredictKernel<method, algorithmFPType, cpu>::compute(const NumericTablePtr & xTable,
                                                                                  const brownboost::interface1::Model * m, NumericTablePtr & rTable,
                                                                                  const brownboost::interface1::Parameter * par)
{
    const size_t nVectors                      = xTable->getNumberOfRows();
    brownboost::interface1::Model * boostModel = const_cast<brownboost::interface1::Model *>(m);
    const size_t nWeakLearners                 = boostModel->getNumberOfWeakLearners();

    services::Status s;
    WriteOnlyColumns<algorithmFPType, cpu> mtR(*rTable, 0, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * r = mtR.get();
    DAAL_ASSERT(r);

    {
        ReadColumns<algorithmFPType, cpu> mtAlpha(*boostModel->getAlpha(), 0, 0, nWeakLearners);
        DAAL_CHECK_BLOCK_STATUS(mtAlpha);
        DAAL_ASSERT(mtAlpha.get());
        DAAL_CHECK_STATUS(s, this->compute(xTable, m, nWeakLearners, mtAlpha.get(), r, par));
    }

    const algorithmFPType error = par->accuracyThreshold;
    const algorithmFPType zero  = (algorithmFPType)0.0;
    if (error != zero)
    {
        algorithmFPType sqrtC    = daal::internal::Math<algorithmFPType, cpu>::sErfInv(algorithmFPType(1.0) - error);
        algorithmFPType invSqrtC = algorithmFPType(1.0) / sqrtC;
        for (size_t j = 0; j < nVectors; j++)
        {
            r[j] *= invSqrtC;
        }
    }
    daal::internal::Math<algorithmFPType, cpu>::vErf(nVectors, r, r);
    return s;
}
} // namespace internal
} // namespace prediction
} // namespace brownboost
} // namespace algorithms
} // namespace daal

#endif
