/* file: batch_normalization_layer.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of batch normalization calculation algorithm and types methods.
//--
*/

#include "batch_normalization_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace batch_normalization
{
namespace interface1
{
/**
 * Constructs the parameters of the batch normalization layer
 * \param[in] alpha             Smoothing factor that is used in population mean and population variance computations
 * \param[in] epsilon           A constant added to the mini-batch variance for numerical stability
 * \param[in] dimension         Index of the dimension for which the normalization is performed
 */
Parameter::Parameter(double alpha, double epsilon, size_t dimension) :
    alpha(alpha), epsilon(epsilon), dimension(dimension)
{}

}// namespace interface1
}// namespace batch_normalization
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
