/* file: average_pooling1d_layer.cpp */
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
//  Implementation of average_pooling1d calculation algorithm and types methods.
//--
*/

#include "average_pooling1d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace average_pooling1d
{
namespace interface1
{
/**
 * Constructs the parameters of average 1D pooling layer
 * \param[in] index        Index of the dimension on which pooling is performed
 * \param[in] kernelSize   Size of 1D subtensor for which the average element is computed
 * \param[in] stride       Interval over the dimension on which the pooling is performed
 * \param[in] padding      Number of data elements to implicitly add to the the dimension
 *                         of the 1D subtensor on which the pooling is performed
 */
Parameter::Parameter(size_t index, size_t kernelSize, size_t stride, size_t padding) :
    layers::pooling1d::Parameter(index, kernelSize, stride, padding)
{}

}// namespace interface1
}// namespace average_pooling1d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
