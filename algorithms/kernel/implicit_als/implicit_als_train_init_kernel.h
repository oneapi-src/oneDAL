/* file: implicit_als_train_init_kernel.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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

using namespace daal::data_management;

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSInitKernelBase : public daal::algorithms::Kernel
{
public:
    services::Status randFactors(size_t nItems, size_t nFactors, algorithmFPType *itemsFactors, int *buffer,
        daal::internal::BaseRNGs<cpu> &baseRng);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class ImplicitALSInitKernel : public ImplicitALSInitKernelBase<algorithmFPType, cpu>
{};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSInitKernel<algorithmFPType, fastCSR, cpu> : public ImplicitALSInitKernelBase<algorithmFPType, cpu>
{
public:
    services::Status compute(const NumericTable *data, NumericTable *itemsFactors, const Parameter *parameter);
};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSInitKernel<algorithmFPType, defaultDense, cpu> : public ImplicitALSInitKernelBase<algorithmFPType, cpu>
{
public:
    services::Status compute(const NumericTable *data, NumericTable *itemsFactors, const Parameter *parameter);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class ImplicitALSInitDistrKernelBase
{};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSInitDistrKernelBase<algorithmFPType, fastCSR, cpu>
{
protected:
    void computeOffsets(size_t nParts, const int *partition, NumericTable **offsets);
    services::Status computeBlocksToLocal(size_t nItems, size_t fullNUsers,
                const size_t *rowIndices, const size_t *colOffsets,
                size_t nParts, const int *partition, NumericTable **blocksToLocal);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class ImplicitALSInitDistrKernel
{};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSInitDistrKernel<algorithmFPType, fastCSR, cpu> :
    public ImplicitALSInitKernelBase<algorithmFPType, cpu>,
    public ImplicitALSInitDistrKernelBase<algorithmFPType, fastCSR, cpu>
{
public:
    services::Status compute(const NumericTable *data, const NumericTable *partitionTable,
                 NumericTable **dataParts, NumericTable **blocksToLocal,
                 NumericTable **userOffsets, NumericTable *partialFactors, const DistributedParameter *parameter);

protected:
    using ImplicitALSInitDistrKernelBase<algorithmFPType, fastCSR, cpu>::computeOffsets;
    using ImplicitALSInitDistrKernelBase<algorithmFPType, fastCSR, cpu>::computeBlocksToLocal;

    services::Status transposeAndSplitCSRTable(size_t nItems, size_t fullNUsers,
                const algorithmFPType *tdata, const size_t *rowIndices, const size_t *colOffsets,
                size_t nParts, const int *partition, NumericTable **dataParts);

    services::Status computePartialFactors(
                size_t nUsers, size_t nItems, size_t nFactors, size_t fullNUsers,
                const algorithmFPType *tdata, const size_t *rowIndices, const size_t *colOffsets, algorithmFPType *partialFactors);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class ImplicitALSInitDistrStep2Kernel
{};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSInitDistrStep2Kernel<algorithmFPType, fastCSR, cpu> :
    public daal::algorithms::Kernel,
    public ImplicitALSInitDistrKernelBase<algorithmFPType, fastCSR, cpu>
{
public:
    services::Status compute(size_t nParts, NumericTable ** dataParts, NumericTable *data, NumericTable **blocksToLocal,
                 NumericTable **itemOffsets);

protected:
    using ImplicitALSInitDistrKernelBase<algorithmFPType, fastCSR, cpu>::computeOffsets;
    using ImplicitALSInitDistrKernelBase<algorithmFPType, fastCSR, cpu>::computeBlocksToLocal;

    services::Status mergeCSRTables(size_t nParts, NumericTable ** dataParts, size_t nRows, algorithmFPType *data,
                 size_t *rowOffsets, size_t *colIndices);
};

}
}
}
}
}
}

#endif
