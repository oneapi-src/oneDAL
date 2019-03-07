/* file: stump_predict_kernel.h */
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
//  Declaration of template function that computes Decision Stump predictions.
//--
*/

#ifndef __STUMP_PREDICT_KERNEL_H__
#define __STUMP_PREDICT_KERNEL_H__

#include "stump_model.h"
#include "stump_predict.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

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

template <Method method, typename algorithmFPtype, CpuType cpu>
class StumpPredictKernel : public Kernel
{
public:
    services::Status compute(const NumericTable *x, const stump::Model *m, NumericTable *r, const Parameter *par);
};

} // namespace daal::algorithms::stump::prediction::internal
}
}
}
} // namespace daal

#endif
