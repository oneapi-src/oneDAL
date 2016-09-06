/* file: dropout_layer_forward_kernel.h */
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
//  Implementation of the forward dropout layer
//--


#ifndef __DROPOUT_LAYER_FORWARD_KERNEL_H__
#define __DROPOUT_LAYER_FORWARD_KERNEL_H__

#include "neural_networks/layers/dropout/dropout_layer.h"
#include "neural_networks/layers/dropout/dropout_layer_types.h"
#include "kernel.h"
#include "service_rng.h"
#include "service_math.h"
#include "numeric_table.h"

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
namespace dropout
{
namespace forward
{
namespace internal
{
/**
 *  \brief Kernel for dropout calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DropoutKernel : public Kernel
{
public:
    void compute(const dropout::forward::Input *input, const dropout::Parameter *parameter,
                 dropout::forward::Result *result);

private:
    const size_t _nRowsInBlock = 5000;
    double retainRatio;
    algorithmFPType inverseRetainRatio;
    int *rngBuffer;

    inline void processBlock(SharedPtr<Tensor> inputTable,
                             size_t nProcessedRows,
                             size_t nRowsInCurrentBlock,
                             SharedPtr<Tensor> resultTable,
                             SharedPtr<Tensor> maskTable,
                             daal::internal::IntRng<int,cpu> &rng);

    inline void processBlockPrediction(SharedPtr<Tensor> inputTable,
                                       size_t nProcessedRows,
                                       size_t nRowsInCurrentBlock,
                                       SharedPtr<Tensor> resultTable,
                                       daal::internal::IntRng<int,cpu> &rng);
};
} // internal
} // forward

} // dropout
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
