/* file: implicit_als_train_init_kernel.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Declaration of structure containing kernels for implicit ALS
//  initialization.
//--
*/

#ifndef __IMPLICIT_ALS_INIT_TRAIN_KERNEL_H__
#define __IMPLICIT_ALS_INIT_TRAIN_KERNEL_H__

#include "implicit_als_training_init_batch.h"
#include "implicit_als_model.h"
#include "kernel.h"

#include "service_micro_table.h"
#include "service_rng.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
namespace internal
{

template <typename algorithmFPType, CpuType cpu>
struct ImplicitALSInitTask;

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSInitKernelBase : public daal::algorithms::Kernel
{
public:
    void randFactors(size_t nItems, size_t nFactors, algorithmFPType *itemsFactors, int *buffer,
                daal::internal::IntRng<int,cpu> &rng);

    friend struct ImplicitALSInitTask<algorithmFPType, cpu>;
};

template <typename algorithmFPType, Method method, CpuType cpu>
class ImplicitALSInitKernel : public ImplicitALSInitKernelBase<algorithmFPType, cpu>
{};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSInitKernel<algorithmFPType, fastCSR, cpu> : public ImplicitALSInitKernelBase<algorithmFPType, cpu>
{
public:
    void compute(const NumericTable *data, NumericTable *itemsFactors, const Parameter *parameter);
};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSInitKernel<algorithmFPType, defaultDense, cpu> : public ImplicitALSInitKernelBase<algorithmFPType, cpu>
{
public:
    void compute(const NumericTable *data, NumericTable *itemsFactors, const Parameter *parameter);
};

template <typename algorithmFPType, CpuType cpu>
struct ImplicitALSInitTask
{
    ImplicitALSInitTask(NumericTable *itemsFactorsTable, ImplicitALSInitKernelBase<algorithmFPType, cpu> *algorithm);

    virtual ~ImplicitALSInitTask();

    daal::internal::BlockMicroTable<algorithmFPType, writeOnly, cpu> mtItemsFactors;
    algorithmFPType *itemsFactors;
};

template <typename algorithmFPType, Method method, CpuType cpu>
class ImplicitALSInitDistrKernel : public ImplicitALSInitKernelBase<algorithmFPType, cpu>
{};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSInitDistrKernel<algorithmFPType, fastCSR, cpu> : public ImplicitALSInitKernelBase<algorithmFPType, cpu>
{
public:
    void compute(const NumericTable *data, NumericTable *partialFactors, const Parameter *parameter);

protected:
    void computePartialFactors(
                size_t nUsers, size_t nItems, size_t nFactors, size_t fullNUsers, size_t seed,
                algorithmFPType *tdata, size_t *rowIndices, size_t *colOffsets, algorithmFPType *partialFactors);
};

}
}
}
}
}
}

#endif
