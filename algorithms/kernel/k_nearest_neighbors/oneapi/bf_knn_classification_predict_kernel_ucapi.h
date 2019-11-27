/* file: bf_knn_classification_predict_kernel_ucapi.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#ifndef __BF_KNN_CLASSIFICATION_PREDICT_KERNEL_UCAPI_H__
#define __BF_KNN_CLASSIFICATION_PREDICT_KERNEL_UCAPI_H__

#include "kernel.h"
#include "numeric_table.h"
#include "bf_knn_classification_model_ucapi_impl.h"
#include "bf_knn_classification_predict_types.h"
#include "oneapi/internal/execution_context.h"

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{

namespace prediction
{
namespace internal
{

using namespace daal::data_management;

template <typename algorithmFpType>
class KNNClassificationPredictKernelUCAPI : public daal::algorithms::Kernel
{
public:
    services::Status compute(const NumericTable * x, const classifier::Model * m, NumericTable * y,
                             const daal::algorithms::Parameter * par);
private:

    void copyPartialSelections(oneapi::internal::ExecutionContextIface& context,
            const oneapi::internal::KernelPtr& kernel_gather_selection,
            oneapi::internal::UniversalBuffer& distances,
            oneapi::internal::UniversalBuffer& categories,
            oneapi::internal::UniversalBuffer& partialDistances,
            oneapi::internal::UniversalBuffer& partialCategories,
            uint32_t curProbeBlockSize,
            uint32_t nK,
            uint32_t& nPart,
            uint32_t totalParts,
            bool bCopyPrevious,
            services::Status* st);

    void initCategories(oneapi::internal::ExecutionContextIface& context,
            const oneapi::internal::KernelPtr& kernel_init_categories,
            const services::Buffer<int>& labels,
            oneapi::internal::UniversalBuffer& categories,
            uint32_t curProbeBlockSize,
            uint32_t curDataBlockSize,
            uint32_t offset,
            services::Status* st);

    void initDistances(oneapi::internal::ExecutionContextIface& context,
            const oneapi::internal::KernelPtr& kernel_init_distances,
            oneapi::internal::UniversalBuffer& dataSq,
            oneapi::internal::UniversalBuffer& distances,
            uint32_t dataBlockSize,
            uint32_t probesBlockSize,
            services::Status* st);

    void computeDistances(oneapi::internal::ExecutionContextIface& context,
            const services::Buffer<algorithmFpType>& data,
            const services::Buffer<algorithmFpType>& probes,
            oneapi::internal::UniversalBuffer& distances,
            uint32_t dataBlockSize,
            uint32_t probeBlockSize,
            uint32_t nFeatures,
            services::Status* st);

    void computeWinners(oneapi::internal::ExecutionContextIface& context,
            const oneapi::internal::KernelPtr& kernel_compute_winners,
            oneapi::internal::UniversalBuffer& categories,
            oneapi::internal::UniversalBuffer& classes,
            uint32_t probesBlockSize,
            uint32_t nK,
            services::Status* st);

    uint32_t _maxWorkItemsPerGroup = 256;

};

} // namespace internal
} // namespace prediction
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
