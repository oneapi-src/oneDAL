/* file: neural_networks_training_model.cpp */
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
//  Implementation of model of the training stage of neural network
//--
*/

#include "neural_networks_weights_and_biases.h"
#include "neural_networks_training_model.h"
#include "neural_networks_training_partial_result.h"
#include "neural_networks_training_result.h"
#include "serialization_utils.h"
#include "daal_strings.h"

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
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_TRAINING_RESULT_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResult, SERIALIZATION_NEURAL_NETWORKS_TRAINING_PARTIAL_RESULT_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResult, SERIALIZATION_NEURAL_NETWORKS_TRAINING_DISTRIBUTED_PARTIAL_RESULT_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_NEURAL_NETWORKS_TRAINING_MODEL_ID);
}

/** \brief Constructor */
Model::Model() : _backwardLayers(new BackwardLayers()), _storeWeightDerivativesInTable(false) { }

Model::Model(services::Status &st) : _storeWeightDerivativesInTable(false)
{
    _backwardLayers.reset(new BackwardLayers());
    if (!_backwardLayers)
        st.add(services::ErrorMemoryAllocationFailed);
}

ModelPtr Model::create(services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL(Model);
}

/**
 * Sets table containing weights and biases of one forward layer of neural network
 * \param[in] idx               Index of the forward layer
 * \param[in] weightsAndBiases  Table containing weights and biases of one forward layer of neural network
 */
services::Status Model::setWeightsAndBiases(size_t idx, const data_management::NumericTablePtr &table)
{
    return _weightsAndBiases->copyFromTable(table, idx);
}

/**
 * Returns the weights and biases of the forward layer of neural network as numeric table
 * \param[in] idx Index of the backward layer
 * \return   Weights and biases derivatives container
 */
data_management::NumericTablePtr Model::getWeightsAndBiases(size_t idx) const
{
    return _weightsAndBiases->copyToTable(idx);
}

/**
 * Returns the weights and biases derivatives of all backward layers of neural network as numeric table
 * \return   Weights and biases derivatives container
 */
data_management::NumericTablePtr Model::getWeightsAndBiasesDerivatives() const
{
    return _weightsAndBiasesDerivatives->copyToTable();
}

/**
 * Returns the weights and biases derivatives of the backward layer of neural network as numeric table
 * \param[in] idx Index of the backward layer
 * \return   Weights and biases derivatives container
 */
data_management::NumericTablePtr Model::getWeightsAndBiasesDerivatives(size_t idx) const
{
    return _weightsAndBiasesDerivatives->copyToTable(idx);
}

}
}
}
}
