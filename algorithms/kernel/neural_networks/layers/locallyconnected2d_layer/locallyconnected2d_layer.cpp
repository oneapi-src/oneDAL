/* file: locallyconnected2d_layer.cpp */
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
//  Implementation of locallyconnected2d calculation algorithm and types methods.
//--
*/

#include "locallyconnected2d_layer_types.h"
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
namespace locallyconnected2d
{
namespace interface1
{
/**
 *  Default constructor
 */
Parameter::Parameter() : groupDimension(1), indices(2, 3), kernelSizes(2, 2), strides(2, 2), paddings(0, 0), nKernels(1), nGroups(1) {}

/**
 * Checks the correctness of the parameter
 */
services::Status Parameter::check() const
{
    DAAL_CHECK_EX(groupDimension <= 3, services::ErrorIncorrectParameter, services::ParameterName, groupDimensionStr());
    DAAL_CHECK_EX(indices.dims[0] <= 3 && indices.dims[1] <= 3, services::ErrorIncorrectParameter, services::ParameterName, indicesStr());
    DAAL_CHECK_EX(indices.dims[0] != indices.dims[1], services::ErrorIncorrectParameter, services::ParameterName, indicesStr());
    DAAL_CHECK_EX(strides.size[0] != 0  && strides.size[1] != 0, services::ErrorIncorrectParameter, services::ParameterName, stridesStr());
    DAAL_CHECK_EX(kernelSizes.size[0] != 0 && kernelSizes.size[1] != 0, services::ErrorIncorrectParameter, services::ParameterName, kernelSizesStr());
    return services::Status();
}

}// namespace interface1
}// namespace locallyconnected2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
