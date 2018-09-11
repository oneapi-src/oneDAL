/* file: boosting_predict_kernel.h */
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
//  Declaration of template function that computes common boosting predictions.
//--
*/

#ifndef __BOOSTING_PREDICT_KERNEL_H__
#define __BOOSTING_PREDICT_KERNEL_H__

#include "kernel.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace boosting
{
namespace prediction
{
namespace internal
{

template <typename algorithmFPType, CpuType cpu>
class BoostingPredictKernel : public Kernel
{
protected:
    services::Status compute(const NumericTablePtr& xTable, const Model *m, size_t nWeakLearners,
                             const algorithmFPType *alpha, algorithmFPType *r, const Parameter *par);
};

}
}
}
}
}

#endif
