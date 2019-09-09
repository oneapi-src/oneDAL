/* file: neural_networks_training_feedforward.cpp */
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
//  Declaration of common functions for using optimizaion solver
//  in feedforward neural network
//--
*/

#include "neural_networks_training_feedforward.h"
#include "daal_strings.h"
#include "services/daal_memory.h"

using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms::neural_networks;
using namespace daal::algorithms::neural_networks::layers;

daal::algorithms::neural_networks::internal::LearnableLayerIndices::LearnableLayerIndices(
                ForwardLayers *forwardLayers)
{
    size_t nLayers = forwardLayers->size();
    nLearnableLayers = 0;
    for(size_t layerId = 0; layerId < nLayers; layerId++)
    {
        forward::Input *forwardInput = forwardLayers->get(layerId)->getLayerInput();
        TensorPtr wTensor = forwardInput->get(forward::weights);
        if (!wTensor) { continue; }
        TensorPtr bTensor = forwardInput->get(forward::biases);
        if (!bTensor) { continue; }
        if (wTensor->getSize() + bTensor->getSize() > 0)
        {
            nLearnableLayers++;
        }
    }
    layerIndices.reset(nLearnableLayers);
    if (!layerIndices.get())
        return;
    size_t iLayer = 0;
    for(size_t layerId = 0; layerId < nLayers; layerId++)
    {
        forward::Input *forwardInput = forwardLayers->get(layerId)->getLayerInput();
        TensorPtr wTensor = forwardInput->get(forward::weights);
        if (!wTensor) { continue; }
        TensorPtr bTensor = forwardInput->get(forward::biases);
        if (!bTensor) { continue; }
        if (wTensor->getSize() + bTensor->getSize() > 0)
        {
            layerIndices[iLayer++] = layerId;
        }
    }
}

daal::algorithms::neural_networks::internal::LearnableLayerIndices::~LearnableLayerIndices()
{}

size_t daal::algorithms::neural_networks::internal::LearnableLayerIndices::nLearnable() const
{
    return nLearnableLayers;
}

size_t daal::algorithms::neural_networks::internal::LearnableLayerIndices::layerIndex(size_t idx) const
{
    return layerIndices[idx];
}

bool daal::algorithms::neural_networks::internal::LearnableLayerIndices::isValid() const
{
    return layerIndices.get();
}
