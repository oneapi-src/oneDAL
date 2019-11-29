/* file: adaboost_predict_impl_v1.i */
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
//  Implementation of Fast method for Ada Boost prediction algorithm.
//--
*/

#ifndef __ADABOOST_PREDICT_IMPL_I_V1__
#define __ADABOOST_PREDICT_IMPL_I_V1__

#include "service_numeric_table.h"
#include "collection.h"
#include "service_math.h"
#include "service_data_utils.h"

namespace daal
{
namespace algorithms
{
namespace adaboost
{
namespace prediction
{
namespace internal
{
using namespace daal::internal;

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status I1AdaBoostPredictKernel<method, algorithmFPType, cpu>::compute(const NumericTablePtr & xTable,
                                                                                const daal::algorithms::adaboost::interface1::Model * boostModel,
                                                                                const NumericTablePtr & rTable,
                                                                                const daal::algorithms::adaboost::interface1::Parameter * par)
{
    const size_t nVectors = xTable->getNumberOfRows();

    const size_t nWeakLearners = boostModel->getNumberOfWeakLearners();
    services::Status s;
    WriteOnlyColumns<algorithmFPType, cpu> mtR(*rTable, 0, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * r = mtR.get();
    DAAL_ASSERT(r);

    {
        ReadColumns<algorithmFPType, cpu> mtAlpha(*boostModel->getAlpha(), 0, 0, nWeakLearners);
        DAAL_CHECK_BLOCK_STATUS(mtAlpha);
        DAAL_ASSERT(mtAlpha.get());
        DAAL_CHECK_STATUS(s, this->compute(xTable, boostModel, nWeakLearners, mtAlpha.get(), r, par));
    }

    const algorithmFPType zero = (algorithmFPType)0.0;
    const algorithmFPType one  = (algorithmFPType)1.0;
    for (size_t j = 0; j < nVectors; j++)
    {
        r[j] = ((r[j] >= zero) ? one : -one);
    }
    return s;
}
} // namespace internal
} // namespace prediction
} // namespace adaboost
} // namespace algorithms
} // namespace daal

#endif
