/* file: adaboost_predict_kernel.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
//  Declaration of template function that computes Ada Boost predictions.
//--
*/

#ifndef __ADABOOST_PREDICT_KERNEL_H__
#define __ADABOOST_PREDICT_KERNEL_H__

#include "adaboost_model.h"
#include "adaboost_predict.h"
#include "kernel.h"
#include "numeric_table.h"
#include "boosting_predict_kernel.h"

using namespace daal::data_management;
using namespace daal::algorithms::boosting::prediction::internal;

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

template <Method method, typename algorithmFPtype, CpuType cpu>
class AdaBoostPredictKernel : public BoostingPredictKernel<algorithmFPtype, cpu>
{
    using BoostingPredictKernel<algorithmFPtype, cpu>::compute;
public:
    services::Status compute(const NumericTablePtr& x, const Model *m, const NumericTablePtr& r, const Parameter *par);
};

} // namespace daal::algorithms::adaboost::prediction::internal
}
}
}
} // namespace daal

#endif
