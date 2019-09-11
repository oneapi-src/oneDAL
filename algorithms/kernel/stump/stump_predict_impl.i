/* file: stump_predict_impl.i */
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
services::Status StumpPredictKernel<method, algorithmFPtype, cpu>::compute(const NumericTable *xTable,
                                                                           const stump::Model *m, NumericTable *rTable,
                                                                           const Parameter *par)
{
    const size_t nVectors  = xTable->getNumberOfRows();
    stump::Model *model = const_cast<stump::Model *>(m);

    const algorithmFPtype splitPoint = model->getSplitValue<algorithmFPtype>();
    const algorithmFPtype leftValue  = model->getLeftSubsetAverage<algorithmFPtype>();
    const algorithmFPtype rightValue = model->getRightSubsetAverage<algorithmFPtype>();

    services::Status s;

    WriteOnlyColumns<algorithmFPtype, cpu> rBD(*rTable, 0, 0, nVectors);
    DAAL_CHECK_STATUS(s, rBD.status());
    algorithmFPtype* r = rBD.get();

    ReadColumns<algorithmFPtype, cpu> xBD(*const_cast<NumericTable *>(xTable), model->getSplitFeature(), 0, nVectors);
    DAAL_CHECK_STATUS(s, xBD.status());
    const algorithmFPtype* x = xBD.get();

    for (size_t i = 0; i < nVectors; i++)
    {
        r[i] = ((x[i] < splitPoint) ? leftValue : rightValue);
    }
    return s;
}

} // namespace daal::algorithms::stump::prediction::internal
}
}
}
} // namespace daal

#endif
