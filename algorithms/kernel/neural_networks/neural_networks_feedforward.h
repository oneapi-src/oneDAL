/* file: neural_networks_feedforward.h */
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
//  Declaration of common functions for feedforward algorithm
//--
*/

#ifndef __NEURAL_NETWORKS_FEEDFORWARD_H__
#define __NEURAL_NETWORKS_FEEDFORWARD_H__

#include "services/error_handling.h"
#include "services/collection.h"
#include "data_management/data/data_collection.h"
#include "algorithms/neural_networks/layers/layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace internal
{

class LastLayerIndices
{
public:
    LastLayerIndices(const services::Collection<layers::NextLayers> *nextLayers,
                     const data_management::KeyValueDataCollectionPtr &tensors);

    virtual ~LastLayerIndices();

    size_t nLast() const;

    size_t layerIndex(size_t idx) const;
    size_t tensorIndex(size_t idx) const;

    bool isValid() const;

protected:
    size_t nLastLayers;
    size_t *buffer;

    size_t *layerIndices;
    size_t *tensorIndices;
};

services::Status processLayerErrors(size_t layerId, const services::Status &layerStatus);

}
}
}
}

#endif
