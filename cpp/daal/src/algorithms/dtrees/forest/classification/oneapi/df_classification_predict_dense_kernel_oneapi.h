/* file: df_classification_predict_dense_kernel_oneapi.h */
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

#ifndef __DF_CLASSIFICATION_PREDICT_DENSE_KERNEL_ONEAPI_H__
#define __DF_CLASSIFICATION_PREDICT_DENSE_KERNEL_ONEAPI_H__

#include "sycl/internal/types.h"
#include "sycl/internal/execution_context.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/decision_forest/decision_forest_classification_predict.h"
#include "src/algorithms/dtrees/forest/classification/df_classification_model_impl.h"
#include "algorithms/decision_forest/decision_forest_classification_model.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::oneapi::internal;
using namespace daal::algorithms::dtrees::internal;
#define _P(...)              \
    do                       \
    {                        \
        printf(__VA_ARGS__); \
        printf("\n");        \
        fflush(0);           \
    } while (0)

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
class PredictKernelOneAPI : public daal::algorithms::Kernel
{
public:
    PredictKernelOneAPI() {};
    PredictKernelOneAPI(const PredictKernelOneAPI &) = delete;
    PredictKernelOneAPI & operator=(const PredictKernelOneAPI &) = delete;
    ~PredictKernelOneAPI() {};

    services::Status buildProgram(ClKernelFactoryIface & factory, const char * programName, const char * programSrc, const char * buildOptions);
    services::Status compute(services::HostAppIface * const pHostApp, const NumericTable * a, const decision_forest::classification::Model * const m,
                             NumericTable * const r, NumericTable * const prob, const size_t nClasses, const VotingMethod votingMethod);
    services::Status predictByAllTrees(const services::Buffer<algorithmFPType> & srcBuffer, const decision_forest::classification::Model * const m,
                                       UniversalBuffer & classHist, size_t nRows, size_t nCols);

    services::Status predictByTreesWeighted(const services::Buffer<algorithmFPType> & srcBuffer, const UniversalBuffer & featureIndexList,
                                            const UniversalBuffer & leftOrClassTypeList, const UniversalBuffer & featureValueList,
                                            const UniversalBuffer & classProba, UniversalBuffer & obsClassHist, algorithmFPType scale, size_t nRows,
                                            size_t nCols, size_t nTrees, size_t maxTreeSize);
    services::Status predictByTreesUnweighted(const services::Buffer<algorithmFPType> & srcBuffer, const UniversalBuffer & featureIndexList,
                                              const UniversalBuffer & leftOrClassTypeList, const UniversalBuffer & featureValueList,
                                              UniversalBuffer & obsClassHist, algorithmFPType scale, size_t nRows, size_t nCols, size_t nTrees,
                                              size_t maxTreeSize);

    services::Status reduceClassHist(const UniversalBuffer & obsClassHist, UniversalBuffer & classHist, size_t nRows, size_t nTrees);
    services::Status determineWinners(const UniversalBuffer & classHist, services::Buffer<algorithmFPType> & resBuffer, size_t nRows);

private:
    const uint32_t _preferableSubGroup = 16; // preferable maximal sub-group size
    const uint32_t _maxLocalSize       = 128;
    const uint32_t _maxGroupsNum       = 256;

    size_t _nClasses;
    size_t _nTreeGroups;
    VotingMethod _votingMethod;

    oneapi::internal::KernelPtr kernelPredictByTreesWeighted;
    oneapi::internal::KernelPtr kernelPredictByTreesUnweighted;
    oneapi::internal::KernelPtr kernelReduceClassHist;
    oneapi::internal::KernelPtr kernelDetermineWinners;
};

} // namespace internal
} // namespace prediction
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
