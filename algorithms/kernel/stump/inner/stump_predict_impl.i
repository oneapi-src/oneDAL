/* file: stump_predict_impl.i */
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
//  Implementation of Fast method for Decision Stump algorithm.
//--
*/

#ifndef __STUMP_PREDICT_IMPL_I__
#define __STUMP_PREDICT_IMPL_I__

#include "algorithm.h"
#include "numeric_table.h"
#include "daal_defines.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace prediction
{
namespace internal
{
using namespace daal::internal;

template <Method method, typename algorithmFPtype, CpuType cpu>
services::Status StumpPredictKernel<method, algorithmFPtype, cpu>::compute(const NumericTable * xTable, const stump::Model * m, NumericTable * rTable,
                                                                           const Parameter * par)
{
    const size_t nVectors = xTable->getNumberOfRows();
    stump::Model * model  = const_cast<stump::Model *>(m);

    const algorithmFPtype splitPoint = model->getSplitValue<algorithmFPtype>();
    const algorithmFPtype leftValue  = model->getLeftSubsetAverage<algorithmFPtype>();
    const algorithmFPtype rightValue = model->getRightSubsetAverage<algorithmFPtype>();

    services::Status s;

    WriteOnlyColumns<algorithmFPtype, cpu> rBD(*rTable, 0, 0, nVectors);
    DAAL_CHECK_STATUS(s, rBD.status());
    algorithmFPtype * r = rBD.get();

    ReadColumns<algorithmFPtype, cpu> xBD(*const_cast<NumericTable *>(xTable), model->getSplitFeature(), 0, nVectors);
    DAAL_CHECK_STATUS(s, xBD.status());
    const algorithmFPtype * x = xBD.get();

    for (size_t i = 0; i < nVectors; i++)
    {
        r[i] = ((x[i] < splitPoint) ? leftValue : rightValue);
    }
    return s;
}

} // namespace internal
} // namespace prediction
} // namespace stump
} // namespace algorithms
} // namespace daal

#endif
