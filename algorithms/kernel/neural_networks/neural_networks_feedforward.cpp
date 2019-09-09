/* file: neural_networks_feedforward.cpp */
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
//  Implementation of common functions for feedforward algorithm
//--
*/

#include "neural_networks_feedforward.h"
#include "services/daal_memory.h"
#include "daal_strings.h"

using namespace daal::services;
using namespace daal::data_management;

daal::algorithms::neural_networks::internal::LastLayerIndices::LastLayerIndices(
            const Collection<layers::NextLayers> *nextLayers,
            const KeyValueDataCollectionPtr &tensors) : layerIndices(NULL), tensorIndices(NULL), buffer(NULL)
{
    size_t nLayers = nextLayers->size();
    nLastLayers = 0;
    for(size_t layerId = 0; layerId < nLayers; layerId++)
    {
        if (nextLayers->get(layerId).size() == 0)
        {
            nLastLayers++;
        }
    }

    buffer = (size_t *)daal_malloc(2 * nLastLayers * sizeof(size_t));
    if (!buffer)
        return;

    layerIndices  = buffer;
    tensorIndices = buffer + nLastLayers;

    for(size_t layerId = 0, iLastLayer = 0; layerId < nLayers; layerId++)
    {
        if (nextLayers->get(layerId).size() == 0)
        {
            layerIndices      [iLastLayer] = layerId;
            tensorIndices[iLastLayer] = layerId;
            iLastLayer++;
        }
    }

    if (nLastLayers == 1 && tensors->getKeyByIndex(0) != layerIndices[0])
    {
        tensorIndices[0] = tensors->getKeyByIndex(0);
    }
}

daal::algorithms::neural_networks::internal::LastLayerIndices::~LastLayerIndices()
{
    daal_free(buffer);
}

size_t daal::algorithms::neural_networks::internal::LastLayerIndices::nLast() const
{
    return nLastLayers;
}

size_t daal::algorithms::neural_networks::internal::LastLayerIndices::layerIndex(size_t idx) const
{
    return layerIndices[idx];
}

size_t daal::algorithms::neural_networks::internal::LastLayerIndices::tensorIndex(size_t idx) const
{
    return tensorIndices[idx];
}

bool daal::algorithms::neural_networks::internal::LastLayerIndices::isValid() const
{
    return (buffer != NULL);
}


Status daal::algorithms::neural_networks::internal::processLayerErrors(size_t layerId, const Status &layerStatus)
{
    if (layerStatus)
        return layerStatus;
    Status s(Error::create(ErrorNeuralNetworkLayerCall, Layer, layerId));
    return (s |= layerStatus);
}
