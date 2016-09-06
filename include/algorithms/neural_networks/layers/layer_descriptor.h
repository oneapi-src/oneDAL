/* file: layer_descriptor.h */
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

/*
//++
//  Implementation of neural network layer descriptor.
//--
*/

#ifndef __LAYER_DESCRIPTOR_H__
#define __LAYER_DESCRIPTOR_H__

#include "algorithms/neural_networks/layers/layer.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace interface1
{
/**
 * @ingroup layers
 * @{
 */
/**
* <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LAYERDESCRIPTOR"></a>
* \brief Class defining descriptor for layer on both forward and backward stages and its parameters
*/
class LayerDescriptor
{
public:
    DAAL_NEW_DELETE();

    /** \brief Constructor */
    LayerDescriptor(): _index(0) {}

    /** \brief Constructor */
    LayerDescriptor(const size_t index_, const layers::LayerIfacePtr &layer_): _index(index_), layer(layer_){}

    /** \brief Constructor */
    LayerDescriptor(const size_t index_, const layers::LayerIfacePtr &layer_, const NextLayers &nextLayers_):
        _index(index_), layer(layer_), nextLayers(nextLayers_) {};

    /** \brief Constructor */
    LayerDescriptor(const LayerDescriptor& o) : _index(o._index), layer(o.layer), nextLayers(o.nextLayers){}

    /**
    * Access to the index of the layer in the network
    * \return Index of the layer in the network
    */
    size_t index() const { return _index; }

    /**
    *  Adds index of a layer to the list of next layers
    *  \param[in] index Index to add
    */
    void addNext(size_t index) { nextLayers.add(index); }

public:
    layers::LayerIfacePtr layer; /*!< Layer algorithm */
    NextLayers nextLayers; /*!< Layers following the current layer in the network */

protected:
    size_t _index; /*!< Index of the layer in the network */
};
/** }@ */
} // interface1
using interface1::LayerDescriptor;

} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
