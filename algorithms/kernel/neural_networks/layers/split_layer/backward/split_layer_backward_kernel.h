/* file: split_layer_backward_kernel.h */
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

//++
//  Declaration of template function that calculate splits.
//--


#ifndef __SPLIT_LAYER_BACKWARD_KERNEL_H__
#define __SPLIT_LAYER_BACKWARD_KERNEL_H__

#include "neural_networks/layers/split/split_layer.h"
#include "neural_networks/layers/split/split_layer_types.h"
#include "kernel.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace split
{
namespace backward
{
namespace internal
{

/**
 *  \brief Kernel for split calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class SplitKernel : public Kernel
{
public:
    void compute(const split::backward::Input *input, const split::Parameter *parameter,
                 split::backward::Result *result);

private:
    const size_t _nRowsInBlock = 5000;

    inline void processBlock(SharedPtr<Tensor> inputTable,
                             size_t nProcessedRows, size_t nRowsInCurrentBlock,
                             SharedPtr<Tensor> resultTable);
    inline void processBlockInit(SharedPtr<Tensor> inputTable,
                                 size_t nProcessedRows, size_t nRowsInCurrentBlock,
                                 SharedPtr<Tensor> resultTable);
};

} // internal
} // backward
} // relu
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
