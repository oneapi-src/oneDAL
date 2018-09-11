/* file: neural_networks_prediction_model.cpp */
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
//  Implementation of model of the prediction stage of neural network
//--
*/

#include "neural_networks_prediction_model.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace prediction
{

/** Default constructor */
Model::Model() : ModelImpl(), _allocatedBatchSize(0) { }

/**
 * Constructs model object for the prediction stage of neural network
 * from the list of forward stages of the layers and the list of connections between the layers
 * \param[in] forwardLayersForModel  List of forward stages of the layers
 * \param[in] nextLayersForModel     List of next layers for each layer with corresponding index
 * \DAAL_DEPRECATED_USE{ Model::create }
 */
Model::Model(const neural_networks::ForwardLayersPtr &forwardLayersForModel,
             const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayersForModel) :
    ModelImpl(forwardLayersForModel, nextLayersForModel), _allocatedBatchSize(0) { }

/**
 * Constructs model object for the prediction stage of neural network from a collection of layer descriptors
 * \param[in] topology  Collection of layer descriptors of every inserted layer
 * \DAAL_DEPRECATED_USE{ Model::create }
 */
Model::Model(const prediction::Topology &topology) : ModelImpl(), _allocatedBatchSize(0)
{
    for(size_t i = 0; i < topology.size(); i++)
    {
        insertLayer(topology[i]);
    }
}

/** Copy constructor */
Model::Model(const Model &model) : ModelImpl(model), _allocatedBatchSize(model._allocatedBatchSize) { }


Model::Model(services::Status &st) : ModelImpl(st), _allocatedBatchSize(0) { }

Model::Model(const neural_networks::ForwardLayersPtr &forwardLayersForModel,
             const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayersForModel,
             services::Status &st) :
    ModelImpl(forwardLayersForModel, nextLayersForModel, st), _allocatedBatchSize(0) { }

Model::Model(const prediction::Topology &topology, services::Status &st) :
    ModelImpl(st), _allocatedBatchSize(0)
{
    for(size_t i = 0; i < topology.size(); i++)
    {
        insertLayer(topology[i]);
    }
}

/**
 * Constructs empty model for the prediction stage of neural network
 * \param[out] stat Status of the model construction
 * \return Empty model for the prediction stage of neural network
 */
ModelPtr Model::create(services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL(Model);
}

/**
 * Constructs model object for the prediction stage of neural network
 * from the list of forward stages of the layers and the list of connections between the layers
 * \param[in] forwardLayersForModel  List of forward stages of the layers
 * \param[in] nextLayersForModel     List of next layers for each layer with corresponding index
 * \param[out] stat                  Status of the model construction
 * \return Model object for the prediction stage of neural network
 */
ModelPtr Model::create(const neural_networks::ForwardLayersPtr &forwardLayersForModel,
                       const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayersForModel,
                       services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(Model, forwardLayersForModel, nextLayersForModel);
}

/**
 * Constructs model object for the prediction stage of neural network from a collection of layer descriptors
 * \param[in] topology  Collection of layer descriptors of every inserted layer
 * \param[out] stat     Status of the model construction
 * \return Model object for the prediction stage of neural network
 */
ModelPtr Model::create(const prediction::Topology &topology, services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(Model, topology);
}


} // namespace prediction
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
