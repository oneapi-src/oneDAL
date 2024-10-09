/* file: naivebayes_train_kernel.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Declaration of template function that computes K-means.
//--
*/

#ifndef __NAIVEBAYES_TRAIN_KERNEL_H__
#define __NAIVEBAYES_TRAIN_KERNEL_H__

#include "algorithms/naive_bayes/multinomial_naive_bayes_model.h"
#include "algorithms/naive_bayes/multinomial_naive_bayes_training_types.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace training
{
namespace internal
{
template <typename intFPtype, Method method, CpuType cpu>
class NaiveBayesBatchTrainKernel : public Kernel
{
public:
    services::Status compute(const NumericTable * data, const NumericTable * labels, Model * r, const Parameter * par);
};

template <typename intFPtype, Method method, CpuType cpu>
class NaiveBayesOnlineTrainKernel : public Kernel
{
public:
    services::Status compute(const NumericTable * data, const NumericTable * labels, PartialModel * r, const Parameter * par);
    services::Status finalizeCompute(PartialModel * p, Model * r, const Parameter * par);
};

template <typename intFPtype, Method method, CpuType cpu>
class NaiveBayesDistributedTrainKernel : public NaiveBayesOnlineTrainKernel<intFPtype, method, cpu>
{
public:
    services::Status merge(size_t na, PartialModel * const * a, PartialModel * r, const Parameter * par);
};

} // namespace internal
} // namespace training
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal

#endif
