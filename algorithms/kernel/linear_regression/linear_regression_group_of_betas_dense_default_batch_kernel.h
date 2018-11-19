/* file: linear_regression_group_of_betas_dense_default_batch_kernel.h */
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
//  Declaration of template class that computes linear regression quality metrics.
//--
*/

#ifndef __LINEAR_REGRESSION_GROUP_OF_BETAS_DENSE_DEFAULT_BATCH_KERNEL_H__
#define __LINEAR_REGRESSION_GROUP_OF_BETAS_DENSE_DEFAULT_BATCH_KERNEL_H__

#include "linear_regression_group_of_betas_types.h"
#include "kernel.h"
#include "numeric_table.h"
#include "algorithm_base_common.h"


namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace quality_metric
{
namespace group_of_betas
{
namespace internal
{

using namespace daal::data_management;

template<Method method, typename algorithmFPType, CpuType cpu>
class GroupOfBetasKernel : public daal::algorithms::Kernel
{
public:
    virtual ~GroupOfBetasKernel() {}
    services::Status compute(const NumericTable* y, const NumericTable* z, const NumericTable* zReducedModel,
        size_t numBeta, size_t numBetaReducedModel, algorithmFPType accuracyThreshold, NumericTable* out[]);

protected:
    static const size_t _nRowsInBlock = 1024;
};

}
}
}
}
}
}

#endif
