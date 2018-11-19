/* file: prelu_layer.cpp */
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
//  Implementation of prelu calculation algorithm and types methods.
//--
*/

#include "prelu_layer_types.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace prelu
{
namespace interface1
{
/**
*  Constructs parameters of the prelu layer
*  \param[in] _dataDimension    Starting data dimension index to apply weight
*  \param[in] _weightsDimension Number of weight dimensions
*/
Parameter::Parameter(const size_t _dataDimension, const size_t _weightsDimension) : dataDimension(_dataDimension),
    weightsDimension(_weightsDimension)
{};

/**
 * Checks the correctness of the parameter
 */
services::Status Parameter::check() const
{
    if(weightsDimension == (size_t)0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ArgumentName, weightsDimensionStr()));
    }
    return services::Status();
}

}// namespace interface1
}// namespace prelu
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
