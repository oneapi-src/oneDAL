/* file: bf_knn_classification_predict_kernel_ucapi.h */
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

#ifndef __BF_KNN_CLASSIFICATION_PREDICT_KERNEL_UCAPI_H__
#define __BF_KNN_CLASSIFICATION_PREDICT_KERNEL_UCAPI_H__

#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "src/algorithms/k_nearest_neighbors/oneapi/bf_knn_classification_model_ucapi_impl.h"
#include "algorithms/k_nearest_neighbors/bf_knn_classification_predict_types.h"
#include "sycl/internal/execution_context.h"

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
    services::Status compute(const NumericTable * x, const classifier::Model * m, NumericTable * y, const daal::algorithms::Parameter * par);

private:
    services::Status copyPartialDistancesAndLabels(oneapi::internal::ExecutionContextIface & context,
                                                   const oneapi::internal::UniversalBuffer & distances,
                                                   const oneapi::internal::UniversalBuffer & labels,
                                                   oneapi::internal::UniversalBuffer & partialDistances,
                                                   oneapi::internal::UniversalBuffer & partialLabels, uint32_t curQueryBlockRows, uint32_t k,
                                                   uint32_t nChunk, uint32_t totalNumberOfChunks);
    services::Status scatterSumOfSquares(oneapi::internal::ExecutionContextIface & context,
                                         const oneapi::internal::UniversalBuffer & dataSumOfSquares, uint32_t dataBlockRowCount,
                                         uint32_t queryBlockRowCount, oneapi::internal::UniversalBuffer & distances);

    services::Status computeDistances(oneapi::internal::ExecutionContextIface & context, const services::Buffer<algorithmFpType> & data,
                                      const services::Buffer<algorithmFpType> & query, oneapi::internal::UniversalBuffer & distances,
                                      uint32_t dataBlockRowCount, uint32_t queryBlockRowCount, uint32_t nFeatures);

    services::Status computeWinners(oneapi::internal::ExecutionContextIface & context, const oneapi::internal::UniversalBuffer & labels,
                                    uint32_t queryBlockRowCount, uint32_t k, oneapi::internal::UniversalBuffer labelsOut);

    services::Status buildProgram(oneapi::internal::ClKernelFactoryIface & kernel_factory);
};

} // namespace internal
} // namespace prediction
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
