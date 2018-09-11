/* file: locallyconnected2d_layer.cpp */
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
