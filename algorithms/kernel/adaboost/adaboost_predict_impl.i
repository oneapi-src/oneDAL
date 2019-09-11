/* file: adaboost_predict_impl.i */
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
//  Implementation of Fast method for Ada Boost prediction algorithm.
//--
*/

#ifndef __ADABOOST_PREDICT_IMPL_I__
#define __ADABOOST_PREDICT_IMPL_I__

#include "service_numeric_table.h"

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
services::Status AdaBoostPredictKernel<method, algorithmFPType, cpu>::compute(const NumericTablePtr& xTable,
    const Model *m, const NumericTablePtr& rTable, const Parameter *par)
{
    const size_t nVectors = xTable->getNumberOfRows();

    Model *boostModel = const_cast<Model *>(m);
    const size_t nWeakLearners = boostModel->getNumberOfWeakLearners();
    services::Status s;
    WriteOnlyColumns<algorithmFPType, cpu> mtR(*rTable, 0, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType *r = mtR.get();
    DAAL_ASSERT(r);

    {
        ReadColumns<algorithmFPType, cpu> mtAlpha(*boostModel->getAlpha(), 0, 0, nWeakLearners);
        DAAL_CHECK_BLOCK_STATUS(mtAlpha);
        DAAL_ASSERT(mtAlpha.get());
        DAAL_CHECK_STATUS(s, this->compute(xTable, m, nWeakLearners, mtAlpha.get(), r, par));
    }

    const algorithmFPType zero = (algorithmFPType)0.0;
    const algorithmFPType one = (algorithmFPType)1.0;
    for(size_t j = 0; j < nVectors; j++)
    {
        r[j] = ((r[j] >= zero) ? one : -one);
    }
    return s;
}

} // namespace daal::algorithms::adaboost::prediction::internal
}
}
}
} // namespace daal

#endif
