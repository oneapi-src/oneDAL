/* file: df_regression_predict_dense_kernel_oneapi.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef __DF_REGRESSION_PREDICT_DENSE_KERNEL_ONEAPI_H__
#define __DF_REGRESSION_PREDICT_DENSE_KERNEL_ONEAPI_H__

#include "sycl/internal/types.h"
#include "sycl/internal/execution_context.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/decision_forest/decision_forest_regression_predict.h"
#include "src/algorithms/dtrees/forest/regression/df_regression_model_impl.h"
#include "algorithms/decision_forest/decision_forest_regression_model.h"
#include "src/services/service_data_utils.h"

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
template <typename algorithmFPType, prediction::Method method>
class PredictKernelOneAPI : public algorithms::Kernel
{
public:
    PredictKernelOneAPI() {};
    PredictKernelOneAPI(const PredictKernelOneAPI &) = delete;
    PredictKernelOneAPI & operator=(const PredictKernelOneAPI &) = delete;
    ~PredictKernelOneAPI() {};

    services::Status buildProgram(oneapi::internal::ClKernelFactoryIface & factory, const char * programName, const char * programSrc);
    services::Status compute(services::HostAppIface * const pHostApp, const data_management::NumericTable * a,
                             const decision_forest::regression::Model * const m, data_management::NumericTable * const r);
    services::Status predictByAllTrees(const services::Buffer<algorithmFPType> & srcBuffer, const decision_forest::regression::Model * const m,
                                       services::Buffer<algorithmFPType> & resObsResponse, size_t nRows, size_t nCols);
    services::Status predictByTreesGroup(const services::Buffer<algorithmFPType> & srcBuffer,
                                         const oneapi::internal::UniversalBuffer & featureIndexList,
                                         const oneapi::internal::UniversalBuffer & leftOrClassTypeList,
                                         const oneapi::internal::UniversalBuffer & featureValueList, oneapi::internal::UniversalBuffer & obsResponses,
                                         size_t nRows, size_t nCols, size_t nTrees, size_t maxTreeSize);
    services::Status reduceResponse(const oneapi::internal::UniversalBuffer & obsResponses, services::Buffer<algorithmFPType> & resObsResponse,
                                    size_t nRows, size_t nTrees, algorithmFPType scale);

private:
    const uint32_t _preferableSubGroup = 16; // preferable maximal sub-group size
    const uint32_t _maxLocalSize       = 128;
    const uint32_t _maxGroupsNum       = 256;

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

    size_t _nTreeGroups;

    oneapi::internal::KernelPtr kernelPredictByTreesGroup;
    oneapi::internal::KernelPtr kernelReduceResponse;
};

} // namespace internal
} // namespace prediction
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
