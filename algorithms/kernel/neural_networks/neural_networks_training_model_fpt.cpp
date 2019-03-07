/* file: neural_networks_training_model_fpt.cpp */
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
//  Implementation of model of the training stage of neural network
//--
*/

#include "neural_networks_training_model.h"
#include "neural_networks_weights_and_biases.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{

template class TensorWeightsAndBiasesDerivatives<DAAL_FPTYPE>;
template class NumericTableWeightsAndBiasesDerivatives<DAAL_FPTYPE>;

namespace training
{

template<typename modelFPType>
services::Status DAAL_EXPORT Model::createWeightsAndBiasesDerivatives()
{
    using namespace services;
    Status s;
    if (_storeWeightDerivativesInTable)
    {
        _weightsAndBiasesDerivatives = NumericTableWeightsAndBiasesDerivatives<modelFPType>::create(
                _forwardLayers, _backwardLayers, &s);
    }
    else
    {
        _weightsAndBiasesDerivatives = TensorWeightsAndBiasesDerivatives<modelFPType>::create(
                _backwardLayers, &s);
    }
    return s;
}

template DAAL_EXPORT services::Status Model::createWeightsAndBiasesDerivatives<DAAL_FPTYPE>();

}
}
}
}
