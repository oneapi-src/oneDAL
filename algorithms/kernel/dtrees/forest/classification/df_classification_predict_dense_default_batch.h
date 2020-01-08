/* file: df_classification_predict_dense_default_batch.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

template <typename algorithmFPType, CpuType cpu>
class PredictClassificationTask;

template <typename algorithmFpType, prediction::Method method, CpuType cpu>
class PredictKernel : public daal::algorithms::Kernel
{
public:

    PredictKernel(): _task(nullptr){};
    ~PredictKernel()
    {
        if(_task)
        {
            delete _task;
        }
    }
    /**
     *  \brief Compute decision forest prediction results.
     *
     *  \param a[in]    Matrix of input variables X
     *  \param m[in]    decision forest model obtained on training stage
     *  \param r[out]   Prediction results
     *  \param par[in]  decision forest algorithm parameters
     */
    services::Status compute(services::HostAppIface * const pHostApp, const NumericTable * a, const decision_forest::classification::Model * const m,
                             NumericTable * const r, NumericTable * const prob, const size_t nClasses, const VotingMethod votingMethod);
    PredictClassificationTask<algorithmFpType, cpu>* _task;
};

} // namespace internal
} // namespace prediction
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
