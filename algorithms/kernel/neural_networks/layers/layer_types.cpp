/* file: layer_types.cpp */
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
//  Implementation of neural_networks layers methods.
//--
*/

#include "layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace interface1
{

/** Default constructor */
Parameter::Parameter() : predictionStage(false), propagateGradient(true),
    weightsInitializer(new initializers::uniform::Batch<>()),
    biasesInitializer(new initializers::uniform::Batch<>()),
    weightsAndBiasesInitialized(false),
    allowInplaceComputation(true)
{}

}// namespace interface1
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
