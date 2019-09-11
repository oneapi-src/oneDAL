/* file: neural_networks_feedforward.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
