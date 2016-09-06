/* file: neural_networks_training_model.cpp */
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
//  Implementation of model of the training stage of neural network
//--
*/

#include "neural_networks_training_model.h"
#include "neural_networks_weights_and_biases.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace training
{

/**
 * Sets table containing weights and biases of one forward layer of neural network
 * \param[in] idx               Index of the forward layer
 * \param[in] weightsAndBiases  Table containing weights and biases of one forward layer of neural network
 */
void Model::setWeightsAndBiases(size_t idx, const data_management::NumericTablePtr &table)
{
    _weightsAndBiases->copyFromTable(table, idx);
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
