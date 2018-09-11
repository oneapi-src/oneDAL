/* file: bernoulli_kernel.h */
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

//++
//  Declaration of template function that calculates bernoulli distribution.
//--

#ifndef __BERNOULLI_KERNEL_H__
#define __BERNOULLI_KERNEL_H__

#include "distributions/bernoulli/bernoulli.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_numeric_table.h"
#include "service_rng.h"
#include "uniform_kernel.h"

using namespace daal::services;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace distributions
{
namespace bernoulli
{
namespace internal
{
/**
 *  \brief Kernel for bernoulli calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class BernoulliKernel : public Kernel
{
public:
    static Status compute(algorithmFPType p, engines::BatchBase &engine, NumericTable *resultTable);
    static Status computeInt(int *resultArray, size_t n, algorithmFPType p, engines::BatchBase &engine);
    static Status computeFPType(NumericTable *resultTable, algorithmFPType p, engines::BatchBase &engine);
};

template<typename algorithmFPType, CpuType cpu>
using BernoulliKernelDefault = BernoulliKernel<algorithmFPType, bernoulli::defaultDense, cpu>;

} // namespace internal
} // namespace bernoulli
} // namespace distributions
} // namespace algorithms
} // namespace daal

#endif
