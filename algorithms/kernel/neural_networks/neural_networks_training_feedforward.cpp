/* file: neural_networks_training_feedforward.cpp */
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
