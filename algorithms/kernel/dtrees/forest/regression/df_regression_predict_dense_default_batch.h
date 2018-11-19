/* file: df_regression_predict_dense_default_batch.h */
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
//  Declaration of template function that computes decision forest
//  prediction results.
//--
*/

#ifndef __DF_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_H__
#define __DF_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_H__

#include "decision_forest_regression_predict.h"
#include "service_memory.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace prediction
{
namespace internal
{

template <typename algorithmFpType, decision_forest::regression::prediction::Method method, CpuType cpu>
class PredictKernel : public daal::algorithms::Kernel
{
public:
    /**
     *  \brief Compute decision forest prediction results.
     *
     *  \param a[in]    Matrix of input variables X
     *  \param m[in]    decision forest model obtained on training stage
     *  \param r[out]   Prediction results
     *  \param par[in]  decision forest algorithm parameters
     */
    services::Status compute(services::HostAppIface* pHostApp, const NumericTable *a, const regression::Model *m, NumericTable *r);
};

} // namespace internal
}
}
}
}
} // namespace daal

#endif
