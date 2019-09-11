/* file: locallyconnected2d_layer_backward.cpp */
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
//  Implementation of locally connected 2d calculation algorithm and types methods.
//--
*/

#include "locallyconnected2d_layer_backward_types.h"
#include "locallyconnected2d_layer_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
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
namespace backward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_LOCALLYCONNECTED2D_BACKWARD_RESULT_ID);
/**
 * Default constructor
 */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/**
 * Returns an input object for backward 2D locally connected layer
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::TensorPtr Input::get(LayerDataId id) const
{
    LayerDataPtr layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::backward::inputFromForward));
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets input for the backward 2D locally connected layer
 * \param[in] id    Identifier of the input  object
 * \param[in] value Input object to set
 */
void Input::set(LayerDataId id, const data_management::TensorPtr &value)
{
    LayerDataPtr layerData = get(layers::backward::inputFromForward);
    (*layerData)[id] = value;
}

/**
 * Checks an input object of the 2D locally connected layer
 * \param[in] parameter %Parameter of layer
 * \param[in] method    Computation method of the layer
 */
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, layers::backward::Input::check(parameter, method));

    const Parameter *param = static_cast<const Parameter *>(parameter);

    data_management::TensorPtr xTensor = get(auxData);
    DAAL_CHECK_STATUS(s, data_management::checkTensor(xTensor.get(), auxDataStr()))
    const services::Collection<size_t> &xDims = xTensor->getDimensions();

    int a = (int)xDims[param->indices.dims[0]] + 2 * param->paddings.size[0] - (int)param->kernelSizes.size[0];
    int b = (int)xDims[param->indices.dims[1]] + 2 * param->paddings.size[1] - (int)param->kernelSizes.size[1];

    DAAL_CHECK(a > 0 || b > 0, ErrorIncorrectParameter);
    DAAL_CHECK_EX(xDims[param->groupDimension] % param->nGroups == 0 || param->nKernels % param->nGroups == 0, ErrorIncorrectParameter,
                  ParameterName, nGroupsStr());

    size_t l3 = (size_t)a / param->strides.size[0] + 1;
    size_t l4 = (size_t)b / param->strides.size[1] + 1;

    services::Collection<size_t> gradDims;
    for(size_t i = 0; i < xDims.size(); i++)
    {
        if(i == param->indices.dims[0]) { gradDims.push_back(l3); }
        else if(i == param->indices.dims[1]) { gradDims.push_back(l4); }
        else if(i == param->groupDimension) { gradDims.push_back(param->nKernels); }
        else { gradDims.push_back( xDims[i] ); }
    }

    services::Collection<size_t> wDims;
    wDims << param->nKernels << l3 << l4 << xDims[param->groupDimension] << param->kernelSizes.size[0] << param->kernelSizes.size[1];

    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::backward::inputGradient).get(), inputGradientStr(), &gradDims));
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxWeights).get(), auxWeightsStr(), &wDims));
    return s;
}
/**
 * Default constructor
 */
Result::Result() : layers::backward::Result() {}

/**
 * Checks the result of the 2D locally connected layer
 * \param[in] input   %Input object of the layer
 * \param[in] par     %Parameter of the layer
 * \param[in] method  Computation method of the layer
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, layers::backward::Result::check(input, par, method));

    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *param = static_cast<const Parameter *>(par);

    if (param->propagateGradient)
    {
        DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::backward::gradient).get(), gradientStr(),
                                          &(algInput->get(auxData)->getDimensions())));
    }
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::backward::weightDerivatives).get(), weightDerivativesStr(),
                                      &(algInput->get(auxWeights)->getDimensions())));

    services::Collection<size_t> bDims;
    getBiasesDims(algInput, param, bDims);

    return data_management::checkTensor(get(layers::backward::biasDerivatives).get(), biasDerivativesStr(), &bDims);
}

void Result::getBiasesDims(const Input *algInput, const Parameter *param, services::Collection<size_t> &bDims) const
{
    data_management::TensorPtr auxDataTensor  = algInput->get(auxData);
    const services::Collection<size_t> &xDims = auxDataTensor->getDimensions();

    size_t l3 = (xDims[param->indices.dims[0]] + 2 * param->paddings.size[0] - param->kernelSizes.size[0]) / param->strides.size[0] + 1;
    size_t l4 = (xDims[param->indices.dims[1]] + 2 * param->paddings.size[1] - param->kernelSizes.size[1]) / param->strides.size[1] + 1;

    bDims << param->nKernels << l3 << l4;
}

}// namespace interface1
}// namespace backward
}// namespace locallyconnected2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
