/* file: neural_networks_training_topology.h */
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

#ifndef __NEURAL_NETWORKS_TRAINING_TOPOLOGY_H__
#define __NEURAL_NETWORKS_TRAINING_TOPOLOGY_H__

#include "algorithms/neural_networks/layers/layer_descriptor.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace training
{
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__TOPOLOGY"></a>
 * \brief Class defining a neural network topology - a set of layers and connection between them -
 *        on the training stage
 */
class Topology: public Base
{
public:
    Topology() {}

    /**
     * Number of layers in the topology
     * \return Size of the collection
     */
    size_t size() const { return _config.size(); }

    /**
    *  Adds an element to the collection of layers and assigns the next available id to it
    *  \param[in] layer Element to add
    * \return Index of the element
    */
    size_t push_back(const layers::LayerIfacePtr &layer)
    {
        size_t id = _config.size();
        _config.push_back(layers::LayerDescriptor(id, layer));
        return id;
    }

    /**
    *  Adds an element to the collection of layers and assigns the next available id to it
    *  \param[in] layer Element to add
    * \return    Index of the element
    */
    size_t add(const layers::LayerIfacePtr &layer)
    {
        return push_back(layer);
    }

    /**
     *  Clears a topology: removes all layer descriptors and sets size to 0
     */
    void clear() { _config.clear(); }

    /**
     * Element access
     * \param[in] index Index of an accessed element
     * \return    Reference to the element
     */
    layers::LayerDescriptor& operator [] (size_t index) { return _config[index]; }

    /**
     * Const element access
     * \param[in] index Index of an accessed element
     * \return    Reference to the element
     */
    const layers::LayerDescriptor& operator [] (size_t index) const { return _config[index]; }

    /**
     * Element access
     * \param[in] index Index of an accessed element
     * \return    Reference to the element
     */
    layers::LayerDescriptor& get(size_t index) { return _config[index]; }

    /**
     * Const element access
     * \param[in] index Index of an accessed element
     * \return    Reference to the element
     */
    const layers::LayerDescriptor& get(size_t index) const { return _config[index]; }

    /**
    * Adds next layer to the given layer
    * \param[in] index Index of the layer to add next layer
    * \param[in] next Index of the next layer
    */
    void addNext(size_t index, size_t next) { _config[index].addNext(next); }

protected:
    services::Collection<layers::LayerDescriptor> _config;
};
}
using interface1::Topology;
typedef services::SharedPtr<Topology> TopologyPtr;
}
}
}
}

#endif
