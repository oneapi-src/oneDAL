/* file: brownboost_predict_impl.i */
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
//  Implementation of Fast method for Brown Boost prediction algorithm.
//--
*/

#ifndef __BROWNBOOST_PREDICT_IMPL_I__
#define __BROWNBOOST_PREDICT_IMPL_I__

#include "service_math.h"
#include "service_micro_table.h"

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

template <Method method, typename algorithmFPType, CpuType cpu>
void BrownBoostPredictKernel<method, algorithmFPType, cpu>::compute(NumericTablePtr xTable,
                                                                    const Model *m, NumericTablePtr rTable,
                                                                    const Parameter *par)
{
    const algorithmFPType zero = (algorithmFPType)0.0;
    const algorithmFPType one  = (algorithmFPType)1.0;

    size_t nVectors  = xTable->getNumberOfRows();

    Model *boostModel = const_cast<Model *>(m);
    size_t nWeakLearners = boostModel->getNumberOfWeakLearners();
    NumericTablePtr alphaTable = boostModel->getAlpha();

    daal::internal::FeatureMicroTable<algorithmFPType, readOnly, cpu> mtAlpha(alphaTable.get());
    daal::internal::FeatureMicroTable<algorithmFPType, writeOnly, cpu> mtR(rTable.get());

    algorithmFPType *alpha = NULL, *r = NULL;
    mtAlpha.getBlockOfColumnValues(0, 0, nWeakLearners, &alpha);
    mtR.getBlockOfColumnValues(0, 0, nVectors, &r);

    this->compute(xTable, m, nWeakLearners, alpha, r, par);
    if(!this->_errors->isEmpty())
    {
        mtAlpha.release();
        mtR.release();
        return;
    }

    Parameter *parameter = const_cast<Parameter *>(par);
    algorithmFPType error   = parameter->accuracyThreshold;
    algorithmFPType invSqrtC;
    if (error != zero)
    {
        algorithmFPType sqrtC = daal::internal::Math<algorithmFPType,cpu>::sErfInv(one - error);
        invSqrtC = one / sqrtC;
        for (size_t j = 0; j < nVectors; j++)
        {
            r[j] *= invSqrtC;
        }
    }
    daal::internal::Math<algorithmFPType,cpu>::vErf(nVectors, r, r);

    mtAlpha.release();
    if(!this->_errors->isEmpty()) { return; }

    mtR.release();
}

} // namespace daal::algorithms::brownboost::prediction::internal
}
}
}
} // namespace daal

#endif
