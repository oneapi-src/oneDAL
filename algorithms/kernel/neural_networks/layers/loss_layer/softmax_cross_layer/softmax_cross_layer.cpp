/* file: softmax_cross_layer.cpp */
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
//  Implementation of softmax cross calculation algorithm and types methods.
//--
*/

#include "softmax_cross_layer_types.h"
#include "daal_strings.h"

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace loss
{
namespace softmax_cross
{
namespace interface1
{
/**
*  Constructs parameters of the softmax cross-entropy layer
*  \param[in] accuracyThreshold_  Value needed to avoid degenerate cases in logarithm computing
*  \param[in] dimension_          Dimension index to calculate softmax cross-entropy
*/
Parameter::Parameter(const double accuracyThreshold_, size_t dimension_) : accuracyThreshold(accuracyThreshold_), dimension(dimension_)
{};

/**
 * Checks the correctness of the parameter
 */
services::Status Parameter::check() const
{
    DAAL_CHECK_EX(accuracyThreshold > 0, services::ErrorIncorrectParameter, services::ParameterName, accuracyThresholdStr());
    return services::Status();
}

}// namespace interface1
}// namespace softmax_cross
}// namespace loss
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
