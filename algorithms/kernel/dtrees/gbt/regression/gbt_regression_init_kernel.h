/* file: gbt_regression_init_kernel.h */
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

/*
//++
//  Declaration of structure containing kernels for distributed
//  gradient boosted trees init.
//--
*/

#ifndef __GBT_REGRESSION_INIT_KERNEL_H__
#define __GBT_REGRESSION_INIT_KERNEL_H__

#include "numeric_table.h"
#include "algorithm_base_common.h"
#include "gbt_regression_init_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace init
{
namespace internal
{

template <typename algorithmFPType, Method method, CpuType cpu>
class RegressionInitStep1LocalKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(const NumericTable *x, const NumericTable *y, const HomogenNumericTable<algorithmFPType> * meanDependentVariable,
                             const HomogenNumericTable<size_t> * numberOfRows, const HomogenNumericTable<algorithmFPType> * binBorders,
                             const HomogenNumericTable<size_t> * binSizes, const Parameter& par);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class RegressionInitStep2MasterKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(size_t nNodes, const DataCollectionPtr localNumberOfRows, const DataCollectionPtr localMeanDepVars,
                             const DataCollectionPtr localBinBorders, const DataCollectionPtr localBinSizes,
                             HomogenNumericTable<algorithmFPType> * ntInitialResponse, const HomogenNumericTable<algorithmFPType> * mergedBinBorders,
                             const HomogenNumericTable<size_t> * binQuantities,
                             DataCollection *dcBinValues,
                             const Parameter& par);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class RegressionInitStep3LocalKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(const HomogenNumericTable<algorithmFPType> *mergedBinBorders, const HomogenNumericTable<size_t> *binQuantities,
                             const NumericTable *x, const HomogenNumericTable<algorithmFPType> *ntInitialResponse,
                             const DistributedPartialResultStep3 *partialResult, const Parameter& par);
};

} // namespace internal
}
}
}
}
} // namespace daal


#endif
