/* file: df_classification_predict_dense_kernel_oneapi.h */
/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
//  Declaration of structure containing kernels for decision forest
//  prediction for GPU for the dense method.
//--
*/

#ifndef __DF_CLASSIFICATION_PREDICT_DENSE_KERNEL_ONEAPI_H__
#define __DF_CLASSIFICATION_PREDICT_DENSE_KERNEL_ONEAPI_H__

#include "services/internal/sycl/types.h"
#include "services/internal/sycl/execution_context.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/decision_forest/decision_forest_classification_predict.h"
#include "src/algorithms/dtrees/forest/classification/df_classification_model_impl.h"
#include "algorithms/decision_forest/decision_forest_classification_model.h"
#include "src/services/service_data_utils.h"

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
template <typename algorithmFPType, prediction::Method method>
class PredictKernelOneAPI : public algorithms::Kernel
{
public:
    PredictKernelOneAPI() : _nClasses(0), _nTreeGroups(0), _votingMethod(VotingMethod::unweighted) {};
    PredictKernelOneAPI(const PredictKernelOneAPI &) = delete;
    PredictKernelOneAPI & operator=(const PredictKernelOneAPI &) = delete;
    ~PredictKernelOneAPI() {};

    services::Status buildProgram(services::internal::sycl::ClKernelFactoryIface & factory, const char * programName, const char * programSrc,
                                  const char * buildOptions);
    services::Status compute(services::HostAppIface * const pHostApp, const data_management::NumericTable * a,
                             const decision_forest::classification::Model * const m, data_management::NumericTable * const r,
                             data_management::NumericTable * const prob, const size_t nClasses, const VotingMethod votingMethod);
    services::Status predictByAllTrees(const services::internal::Buffer<algorithmFPType> & srcBuffer,
                                       const decision_forest::classification::Model * const m, services::internal::sycl::UniversalBuffer & classHist,
                                       size_t nRows, size_t nCols);

    services::Status predictByTreesWeighted(const services::internal::Buffer<algorithmFPType> & srcBuffer,
                                            const services::internal::sycl::UniversalBuffer & featureIndexList,
                                            const services::internal::sycl::UniversalBuffer & leftOrClassTypeList,
                                            const services::internal::sycl::UniversalBuffer & featureValueList,
                                            const services::internal::sycl::UniversalBuffer & classProba,
                                            services::internal::sycl::UniversalBuffer & obsClassHist, algorithmFPType scale, size_t nRows,
                                            size_t nCols, size_t nTrees, size_t maxTreeSize);
    services::Status predictByTreesUnweighted(const services::internal::Buffer<algorithmFPType> & srcBuffer,
                                              const services::internal::sycl::UniversalBuffer & featureIndexList,
                                              const services::internal::sycl::UniversalBuffer & leftOrClassTypeList,
                                              const services::internal::sycl::UniversalBuffer & featureValueList,
                                              services::internal::sycl::UniversalBuffer & obsClassHist, algorithmFPType scale, size_t nRows,
                                              size_t nCols, size_t nTrees, size_t maxTreeSize);

    services::Status reduceClassHist(const services::internal::sycl::UniversalBuffer & obsClassHist,
                                     services::internal::sycl::UniversalBuffer & classHist, size_t nRows, size_t nTrees);
    services::Status determineWinners(const services::internal::sycl::UniversalBuffer & classHist,
                                      services::internal::Buffer<algorithmFPType> & resBuffer, size_t nRows);

private:
    const size_t _preferableSubGroup = 16; // preferable maximal sub-group size
    const size_t _maxLocalSize       = 128;
    const size_t _maxGroupsNum       = 256;

    // following constants showed best performance on benchmark's datasets
    const size_t _nRowsLarge  = 500000;
    const size_t _nRowsMedium = 100000;

    const size_t _nRowsBlocksForLarge  = 16;
    const size_t _nRowsBlocksForMedium = 8;

    const size_t _nTreesLarge  = 192;
    const size_t _nTreesMedium = 48;
    const size_t _nTreesSmall  = 12;

    const size_t _nTreeGroupsForLarge  = 128;
    const size_t _nTreeGroupsForMedium = 32;
    const size_t _nTreeGroupsForSmall  = 16;
    const size_t _nTreeGroupsMin       = 8;

    static constexpr size_t _int32max = static_cast<size_t>(services::internal::MaxVal<int32_t>::get());

    size_t _nClasses;
    size_t _nTreeGroups;
    VotingMethod _votingMethod;

    services::internal::sycl::KernelPtr kernelPredictByTreesWeighted;
    services::internal::sycl::KernelPtr kernelPredictByTreesUnweighted;
    services::internal::sycl::KernelPtr kernelReduceClassHist;
    services::internal::sycl::KernelPtr kernelDetermineWinners;
};

} // namespace internal
} // namespace prediction
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
