/* file: transposed_conv2d_layer_backward.cpp */
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
//  Implementation of transposed convolution2d calculation algorithm and types methods.
//--
*/

#include "transposed_conv2d_layer_backward_types.h"
#include "transposed_conv2d_layer_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace transposed_conv2d
{
namespace backward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_TRANSPOSED_CONV2D_BACKWARD_RESULT_ID);
/**
 * Default constructor
 */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/**
 * Returns an input object for backward 2D transposed convolution layer
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::TensorPtr Input::get(LayerDataId id) const
{
    layers::LayerDataPtr layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::backward::inputFromForward));
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets input for the backward 2D transposed convolution layer
 * \param[in] id    Identifier of the input  object
 * \param[in] value Input object to set
 */
void Input::set(LayerDataId id, const data_management::TensorPtr &value)
{
    layers::LayerDataPtr layerData = get(layers::backward::inputFromForward);
    (*layerData)[id] = value;
}

/**
 * Checks an input object of the 2D transposed convolution layer
 * \param[in] parameter %Parameter of layer
 * \param[in] method    Computation method of the layer
 *
 * \return Status of computations
 */
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, layers::backward::Input::check(parameter, method));

    data_management::TensorPtr xTensor = get(auxData);
    DAAL_CHECK_STATUS(s, data_management::checkTensor(xTensor.get(), auxDataStr()));

    const Parameter *par = static_cast<const Parameter *>(parameter);

    services::Collection<size_t> gradDims;
    size_t c1 = par->valueSizes.size[0];
    size_t c2 = par->valueSizes.size[1];

    const services::Collection<size_t> &xDims = xTensor->getDimensions();
    if(c1 == 0 && c2 == 0)
    {
        c1 = par->strides.size[0] * xDims[par->indices.dims[0]] + par->kernelSizes.size[0] - par->strides.size[0] - 2 * par->paddings.size[0];
        c2 = par->strides.size[1] * xDims[par->indices.dims[1]] + par->kernelSizes.size[1] - par->strides.size[1] - 2 * par->paddings.size[1];
    }
    for(size_t i = 0; i < xDims.size(); i++)
    {
        if(i == par->indices.dims[0]) { gradDims.push_back(c1); }
        else if(i == par->indices.dims[1]) { gradDims.push_back(c2); }
        else if(i == par->groupDimension) { gradDims.push_back(par->nKernels); }
        else { gradDims.push_back( xDims[i] ); }
    }

    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::backward::inputGradient).get(), inputGradientStr(), &gradDims));

    services::Collection<size_t> wDims;
    if (par->nGroups > 1) {
        wDims.push_back(par->nGroups);
    }
    wDims.push_back(xTensor->getDimensionSize(par->groupDimension) / (par->nGroups));
    wDims.push_back((par->nKernels) / (par->nGroups));
    wDims.push_back(par->kernelSizes.size[0]);
    wDims.push_back(par->kernelSizes.size[1]);
    return data_management::checkTensor(get(auxWeights).get(), auxWeightsStr(), &wDims);
}

/**
 * Default constructor
 */
Result::Result() : layers::backward::Result() {}

/**
 * Checks the result of the 2D transposed convolution layer
 * \param[in] input   %Input object of the layer
 * \param[in] par     %Parameter of the layer
 * \param[in] method  Computation method of the layer
 *
 * \return Status of computations
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, layers::backward::Result::check(input, par, method));

    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *param = static_cast<const Parameter *>(par);

    if (param->propagateGradient)
    {
        DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::backward::gradient).get(), gradientStr(), &(algInput->get(auxData)->getDimensions())));
    }
    DAAL_CHECK_STATUS(s, data_management::checkTensor(
        get(layers::backward::weightDerivatives).get(), weightDerivativesStr(), &(algInput->get(auxWeights)->getDimensions())));

    services::Collection<size_t> bDims;
    bDims.push_back(param->nKernels);

    return data_management::checkTensor(get(layers::backward::biasDerivatives).get(), biasDerivativesStr(), &bDims);
}

}// namespace interface1
}// namespace forward
}// namespace transposed_conv2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
