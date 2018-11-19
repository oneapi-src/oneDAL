/* file: batch_normalization_layer.cpp */
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
