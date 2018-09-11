/* file: df_regression_predict_dense_default_batch_impl.i */
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
//  Common functions for decision forest regression predictions calculation
//--
*/

#ifndef __DF_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __DF_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "algorithm.h"
#include "numeric_table.h"
#include "df_regression_predict_dense_default_batch.h"
#include "threading.h"
#include "daal_defines.h"
#include "df_regression_model_impl.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"
#include "service_memory.h"
#include "dtrees_regression_predict_dense_default_impl.i"
#include "service_algo_utils.h"

using namespace daal::internal;
using namespace daal::services::internal;

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

//////////////////////////////////////////////////////////////////////////////////////////
// PredictRegressionTask
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictRegressionTask : public dtrees::regression::prediction::internal::PredictRegressionTaskBase<algorithmFPType, cpu>
{
public:
    typedef dtrees::regression::prediction::internal::PredictRegressionTaskBase<algorithmFPType, cpu> super;
    PredictRegressionTask(const NumericTable *x, NumericTable *y): super(x, y){}

    services::Status run(const decision_forest::regression::internal::ModelImpl* m, services::HostAppIface* pHostApp);
};

//////////////////////////////////////////////////////////////////////////////////////////
// PredictKernel
//////////////////////////////////////////////////////////////////////////////////////////
template<typename algorithmFPType, prediction::Method method, CpuType cpu>
services::Status PredictKernel<algorithmFPType, method, cpu>::compute(services::HostAppIface* pHostApp, const NumericTable *x,
    const regression::Model *m, NumericTable *r)
{
    const daal::algorithms::decision_forest::regression::internal::ModelImpl* pModel =
        static_cast<const daal::algorithms::decision_forest::regression::internal::ModelImpl*>(m);
    PredictRegressionTask<algorithmFPType, cpu> task(x, r);
    return task.run(pModel, pHostApp);
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictRegressionTask<algorithmFPType, cpu>::run(const decision_forest::regression::internal::ModelImpl* m,
    services::HostAppIface* pHostApp)
{
    DAAL_CHECK_MALLOC(this->_featHelper.init(*this->_data));
    const auto nTreesTotal = m->size();
    this->_aTree.reset(nTreesTotal);
    DAAL_CHECK_MALLOC(this->_aTree.get());
    for(size_t i = 0; i < nTreesTotal; ++i)
        this->_aTree[i] = m->at(i);
    const algorithmFPType div = algorithmFPType(1) / algorithmFPType(nTreesTotal);
    return super::run(pHostApp, div);
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace regression */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif
