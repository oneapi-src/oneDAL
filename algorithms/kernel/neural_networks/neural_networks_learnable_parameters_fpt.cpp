/* file: neural_networks_learnable_parameters_fpt.cpp */
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
//  Implementation of classes for storing learnable parameters of neural network
//--
*/

#include "neural_networks_learnable_parameters.h"
#include "neural_networks_weights_and_biases.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{

template class TensorWeightsAndBiases<DAAL_FPTYPE>;
template class NumericTableWeightsAndBiases<DAAL_FPTYPE>;

template<typename modelFPType>
services::Status DAAL_EXPORT ModelImpl::createWeightsAndBiases(bool checkAllocation)
{
    using namespace services;
    if (_weightsAndBiasesCreated) { return services::Status(); }
    services::Status s;

    if (checkAllocation)
    {
        DAAL_CHECK_STATUS(s, checkWeightsAndBiasesAllocation());
    }

    if (_storeWeightsInTable)
    {
        _weightsAndBiases = NumericTableWeightsAndBiases<modelFPType>::create(_forwardLayers, &s);
    }
    else
    {
        _weightsAndBiases = TensorWeightsAndBiases<modelFPType>::create(_forwardLayers, &s);
    }
    if (s)
        _weightsAndBiasesCreated = true;

    return s;
}

template DAAL_EXPORT services::Status ModelImpl::createWeightsAndBiases<DAAL_FPTYPE>(bool checkAllocation);

}
}
}
