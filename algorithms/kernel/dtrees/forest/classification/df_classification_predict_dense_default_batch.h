/* file: df_classification_predict_dense_default_batch.h */
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

#ifndef __DF_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_H__
#define __DF_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_H__

#include "decision_forest_classification_predict.h"
#include "service_memory.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_algo_utils.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace prediction
{
namespace internal
{

template <typename algorithmFpType, prediction::Method method, CpuType cpu>
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
    services::Status compute(services::HostAppIface* pHostApp, const NumericTable *a,
        const decision_forest::classification::Model *m, NumericTable *r, size_t nClasses);
};

} // namespace internal
}
}
}
}
} // namespace daal

#endif
