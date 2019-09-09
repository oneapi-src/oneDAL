/* file: batch_normalization_layer_forward_fpt.cpp */
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
//  Implementation of batch_normalization calculation algorithm and types methods.
//--
*/

#include "batch_normalization_layer_forward_types.h"
#include "batch_normalization_layer_types.h"
#include "daal_strings.h"

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
namespace forward
{
namespace interface1
{
/**
 * Allocates memory to store input objects of forward batch normalization layer
 * \param[in] parameter %Parameter of forward batch normalization layer
 * \param[in] method    Computation method for the layer
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Input::allocate(const daal::algorithms::Parameter *parameter, const int method)
{
    const Parameter *param =  static_cast<const Parameter *>(parameter);
    services::Status s;
    if (!get(layers::forward::weights))
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::forward::weights, getWeightsSizes(param));
    }

    if (!get(layers::forward::biases))
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::forward::biases, getBiasesSizes(param));
    }
    return s;
}
/**
 * Allocates memory to store the result of the forward batch normalization layer
 * \param[in] input Pointer to an object containing the input data
 * \param[in] parameter %Parameter of the forward batch normalization layer
 * \param[in] method Computation method for the layer
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status s;
    const Input *in = static_cast<const Input *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    if (!get(layers::forward::value))
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::forward::value, in->get(layers::forward::data)->getDimensions());
    }
    if (!get(layers::forward::resultForBackward))
    {
        set(layers::forward::resultForBackward, LayerDataPtr(new LayerData()));
    }

    const layers::Parameter *par = static_cast<const layers::Parameter * >(parameter);
    if(!par->predictionStage)
    {
        s |= setResultForBackward(input);

        size_t dimension = algParameter->dimension;
        size_t dimensionSize = in->get(layers::forward::data)->getDimensionSize(dimension);
        services::Collection<size_t> auxDims(1);
        auxDims[0] = dimensionSize;

        if (!get(auxMean))
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(s, auxMean, auxDims);
        }
        if (!get(auxStandardDeviation))
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(s, auxStandardDeviation, auxDims);
        }
        if (!get(auxPopulationMean))
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(s, auxPopulationMean, auxDims);
        }
        if (!get(auxPopulationVariance))
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(s, auxPopulationVariance, auxDims);
        }
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT services::Status Input::allocate<DAAL_FPTYPE>(const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace forward
}// namespace batch_normalization
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
