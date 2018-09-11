/* file: neural_networks_feedforward.cpp */
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
