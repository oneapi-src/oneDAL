/* file: softmax_cross_layer.cpp */
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
