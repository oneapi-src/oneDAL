/* file: elu_common.h */
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

#ifndef __ELU_LAYER_COMMON_H__
#define __ELU_LAYER_COMMON_H__

#include <stdint.h>

#include "kernel.h"
#include "threading.h"

#include "service_unique_ptr.h"
#include "service_mkl_tensor.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace elu
{
namespace internal
{

template<typename T, CpuType cpu>
class ScalableTlsBuffer
{
private:
    daal::internal::UniquePtr<daal::tls<T *>, cpu> _tls;

public:
    explicit ScalableTlsBuffer(size_t size)
    {
        _tls.reset(new daal::tls<T *>([ = ]() -> T * {
            return services::internal::service_scalable_malloc<T, cpu>(size);
        }));
    }

    ~ScalableTlsBuffer()
    {
        _tls->reduce([ = ](T *ptr) {
            services::internal::service_scalable_free<T, cpu>(ptr);
        });
    }

    inline T *local()
    {
        return _tls->local();
    }
};

/**
 * Type of indices
 */
typedef uint16_t BlockSizeType;

/**
 *  Block size was determined heuristically.
 *  If you are changin block size make sure size of type
 *  uint16_t is enough to store indices in the 'computeBlock' method.
 *  Note that block size on the backward layer must be the same.
 */
template<typename algorithmFPType, CpuType cpu>
BlockSizeType getMaxBlockSize() { return (BlockSizeType)512; }

/**
 * Body must be the functor of type void(size_t offset, size_t blockSize)
 */
template<typename algorithmFPType, CpuType cpu, typename Body>
void computeThreaded(size_t dataSize, const Body &body)
{
    const auto MAX_BLOCK_SIZE = getMaxBlockSize<algorithmFPType, cpu>();

    const size_t tailBlockSize    = dataSize % MAX_BLOCK_SIZE;
    const size_t regularBlockSize = MAX_BLOCK_SIZE;
    const size_t blocksNumber     = dataSize / regularBlockSize + (size_t)(tailBlockSize > 0);

    daal::threader_for(blocksNumber, blocksNumber, [ & ](int blockIndex)
    {
        const size_t blockSize = (blockIndex < blocksNumber - 1) || (tailBlockSize == 0)
                                 ? regularBlockSize : tailBlockSize;

        const size_t offset = regularBlockSize * blockIndex;

        body(offset, blockSize);
    });
}

/**
 * Returns true if computations can be performed in MKL DNN layout
 */
template<typename algorithmFPType, CpuType cpu>
bool canComputeInMklLayout(const data_management::Tensor &dataTensor,
                           const data_management::Tensor &valueTensor)
{
    using namespace daal::internal;

    return canCastToMklTensor<algorithmFPType>(dataTensor) &&
           canCastToMklTensor<algorithmFPType>(valueTensor);
}

template<typename algorithmFPType, CpuType cpu>
bool canComputeInMklLayout(const data_management::Tensor &dataTensor,
                           const data_management::Tensor &valueTensor,
                           const data_management::Tensor &auxTensor)
{
    using namespace daal::internal;

    return canCastToMklTensor<algorithmFPType>(dataTensor) &&
           canCastToMklTensor<algorithmFPType>(valueTensor) &&
           canCastToMklTensor<algorithmFPType>(auxTensor);
}

} // namespace internal
} // namespace elu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
