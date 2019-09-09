/* file: xavier_initializer_misc_fpt.cpp */
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

#include "xavier_initializer_misc.h"

#include "algorithms/neural_networks/layers/convolution2d/convolution2d_layer_forward.h"
#include "algorithms/neural_networks/layers/fullyconnected/fullyconnected_layer_forward.h"

using namespace daal::services;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace xavier
{
namespace internal
{

#define CONVOLUTION_2D_WEIGHTS_SIZE         4
#define CONVOLUTION_2D_GROUPED_WEIGHTS_SIZE 5

template<typename algorithmFPType>
services::Status getFanInAndFanOut(const XavierInitializerTaskDescriptor &desc,
                                   size_t &fanIn, size_t &fanOut)
{
    layers::forward::LayerIface *layer = desc.layer;
    const Collection<size_t> &shape    = desc.result->getDimensions();

    TensorPtr weightsTensor = layer->getLayerInput()->get(layers::forward::weights);
    bool isWeightsTensor    = weightsTensor.get() == desc.result;

    auto convolutionLayer = dynamic_cast<layers::convolution2d::forward::Batch<algorithmFPType>*>(layer);
    if (convolutionLayer && isWeightsTensor)
    {
        DAAL_CHECK(shape.size() == CONVOLUTION_2D_WEIGHTS_SIZE ||
                   shape.size() == CONVOLUTION_2D_GROUPED_WEIGHTS_SIZE,
                   ErrorIncorrectSizeOfDimensionInTensor);

        size_t offset = 0;
        bool hasGroups = (shape.size() == CONVOLUTION_2D_GROUPED_WEIGHTS_SIZE);
        if (hasGroups) { offset = 1; }

        fanIn  = shape[offset + 1] * shape[offset + 2] * shape[offset + 3];
        fanOut = shape[offset + 0] * shape[offset + 2] * shape[offset + 3];

        return services::Status();
    }

    auto fullyconnectedLayer = dynamic_cast<layers::fullyconnected::forward::Batch<algorithmFPType>*>(layer);
    if (fullyconnectedLayer && isWeightsTensor)
    {
        fanOut = shape[0];
        fanIn  = desc.result->getSize(1, shape.size() - 1);

        return services::Status();
    }

    const Collection<size_t> &inputShape  = layer->getLayerInput()->get(layers::forward::data)->getDimensions();
    const Collection<size_t> &outputShape = layer->getLayerResult()->getValueSize(
        layer->getLayerInput()->get(layers::forward::data)->getDimensions(),
        layer->getLayerParameter(),
        layer->getMethod()
    );

    fanIn = 1;
    for (size_t i = 1; i < inputShape.size(); i++)
    { fanIn *= inputShape[i]; }

    fanOut = 1;
    for (size_t i = 1; i < outputShape.size(); i++)
    { fanOut *= outputShape[i]; }

    return services::Status();
}

template DAAL_EXPORT services::Status getFanInAndFanOut<DAAL_FPTYPE>(
    const XavierInitializerTaskDescriptor &desc, size_t &fanIn, size_t &fanOut);

} // internal
} // xavier
} // initializers
} // neural_networks
} // algorithms
} // daal
