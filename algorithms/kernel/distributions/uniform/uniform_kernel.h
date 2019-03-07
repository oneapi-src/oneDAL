/* file: uniform_kernel.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Declaration of template function that calculates uniform distribution.
//--

#ifndef __UNIFORM_KERNEL_H__
#define __UNIFORM_KERNEL_H__

#include "distributions/uniform/uniform.h"

#include "kernel.h"
#include "numeric_table.h"
#include "engine_batch_impl.h"

#include "service_rng.h"
#include "service_unique_ptr.h"
#include "service_numeric_table.h"

using namespace daal::services;
using namespace daal::internal;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace distributions
{
namespace uniform
{
namespace internal
{
/**
 *  \brief Kernel for uniform calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class UniformKernel : public Kernel
{
public:
    static Status compute(const uniform::Parameter<algorithmFPType> &parameter, engines::BatchBase &engine, NumericTable *resultTable);
    static Status compute(const uniform::Parameter<algorithmFPType> &parameter, engines::BatchBase &engine, size_t n, algorithmFPType *resultArray);
    static Status compute(const uniform::Parameter<algorithmFPType> &parameter, UniquePtr<engines::internal::BatchBaseImpl, cpu> &enginePtr, size_t n, algorithmFPType *resultArray);
    static Status compute(algorithmFPType a, algorithmFPType b, engines::BatchBase &engine, size_t n, algorithmFPType *resultArray);
    static Status compute(algorithmFPType a, algorithmFPType b, engines::internal::BatchBaseImpl &engine, size_t n, algorithmFPType *resultArray);
};

template<typename algorithmFPType, CpuType cpu>
using UniformKernelDefault = UniformKernel<algorithmFPType, uniform::defaultDense, cpu>;

} // namespace internal
} // namespace uniform
} // namespace distributions
} // namespace algorithms
} // namespace daal

#endif
