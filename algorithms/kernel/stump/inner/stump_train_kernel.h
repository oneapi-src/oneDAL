/* file: stump_train_kernel.h */
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
//  Declaration of template function that trains Decision Stump.
//--
*/

#ifndef __STUMP_TRAIN_KERNEL_H__
#define __STUMP_TRAIN_KERNEL_H__

#include "stump_training_types.h"
#include "stump_model.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace training
{
namespace internal
{
template <Method method, typename algorithmFPtype, CpuType cpu>
class StumpTrainKernel : public Kernel
{
public:
    services::Status compute(size_t n, const NumericTable * const * a, Model * r, const Parameter * par);

private:
    void StumpQSort(size_t n, algorithmFPtype * x, algorithmFPtype * w, algorithmFPtype * z);

    services::Status stumpRegressionOrdered(size_t nVectors, const algorithmFPtype * x, const algorithmFPtype * w, const algorithmFPtype * z,
                                            algorithmFPtype sumW, algorithmFPtype sumM, algorithmFPtype sumS, algorithmFPtype & minS,
                                            algorithmFPtype & splitPoint, algorithmFPtype & lMean, algorithmFPtype & rMean);

    services::Status stumpRegressionCategorical(size_t n, size_t nCategories, const int * x, const algorithmFPtype * w, const algorithmFPtype * z,
                                                algorithmFPtype sumW, algorithmFPtype sumM, algorithmFPtype sumS, algorithmFPtype & minS,
                                                algorithmFPtype & splitPoint, algorithmFPtype & lMean, algorithmFPtype & rMean);

    void computeSums(size_t n, const algorithmFPtype * w, const algorithmFPtype * z, algorithmFPtype & sumW, algorithmFPtype & sumM,
                     algorithmFPtype & sumS);

    services::Status doStumpRegression(size_t n, size_t dim, const NumericTable * x, const algorithmFPtype * w, const algorithmFPtype * z,
                                       size_t & splitFeature, algorithmFPtype & splitPoint, algorithmFPtype & leftValue,
                                       algorithmFPtype & rightValue);
};

} // namespace internal
} // namespace training
} // namespace stump
} // namespace algorithms
} // namespace daal

#endif
