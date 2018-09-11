/* file: layer_forward_descriptor.h */
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
//  Implementation of neural network forward layer descriptor.
//--
*/

#ifndef __LAYER_FORWARD_DESCRIPTOR_H__
#define __LAYER_FORWARD_DESCRIPTOR_H__

#include "algorithms/neural_networks/layers/layer_forward.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for training and prediction using neural network
 */
namespace neural_networks
{
/**
 * \brief Contains classes for neural network layers
 */
namespace layers
{
/**
 * \brief Contains classes for the forward stage of the neural network layer
 */
namespace forward
{
namespace interface1
{
/**
 * @ingroup layers_forward
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARD__LAYERDESCRIPTOR"></a>
 * \brief Class defining descriptor for layer on forward stage
 */
class LayerDescriptor
{
public:
    DAAL_NEW_DELETE();

    /** Constructs an empty layer descriptor with zero index */
    LayerDescriptor(): _index(0) {}

    /**
     * Constructs a layer descriptor from the provided layer object and an index to be assigned to this object,
     * and the collection of indices of the layers following the current layer in the network
     * \param[in] index_        Index of the neural network layer
     * \param[in] layer_        Neural network layer
     * \param[in] nextLayers_   Layers following the current layer in the network
     */
    LayerDescriptor(const size_t index_, const forward::LayerIfacePtr &layer_, const NextLayers &nextLayers_) :
        _index(index_), _layer(layer_), _nextLayers(nextLayers_) {}

    /**
     * Constructs a layer descriptor from the provided layer object and an index to be assigned to this object
     * \param[in] index_        Index of the neural network layer
     * \param[in] layer_        Neural network layer
     */
    LayerDescriptor(const size_t index_, const forward::LayerIfacePtr &layer_) : _index(index_), _layer(layer_) {}

    /** Constructor */
    LayerDescriptor(const LayerDescriptor& other): _index(other._index), _layer(other._layer), _nextLayers(other._nextLayers) {}

    /**
     * Access to the index of the layer in the network
     * \return Index of the layer in the network
     */
    size_t index() const { return _index; }

    /**
     * Adds index of a layer to the list of next layers
     * \param[in] index Index to add
     *
     * \return Status of computations
     */
    services::Status addNext(size_t index)
    {
        _nextLayers.add(index);
        return services::Status();
    }

    /**
     * Access to the layer in the network
     * \return The layer in the network
     */
    const forward::LayerIfacePtr & layer() const { return _layer; }

    /**
     * Access to the collection of indices of the layers following the current layer in the network
     * \return The collection of indices of the layers following the current layer in the network
     */
    const NextLayers & nextLayers() const { return _nextLayers; }

private:
    forward::LayerIfacePtr _layer; /*!< Layer algorithm */
    NextLayers _nextLayers;        /*!< Layers following the current layer in the network */
    size_t _index;                 /*!< Index of the layer in the network */
};
/** }@ */
} // namespace interface1
using interface1::LayerDescriptor;
} // namespace forward
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
