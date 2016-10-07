/* file: spatial_pooling2d_layer_forward.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of spatial pooling2d calculation algorithm and types methods.
//--
*/

#include "spatial_pooling2d_layer_types.h"
#include "spatial_pooling2d_layer_forward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace spatial_pooling2d
{
namespace forward
{
namespace interface1
{
/**
 * Default constructor
 */
Input::Input() {}

/**
 * Returns dimensions of weights tensor
 * \return Dimensions of weights tensor
 */
const services::Collection<size_t> Input::getWeightsSizes(const layers::Parameter *parameter) const
{
    return services::Collection<size_t>();
}

/**
 * Returns dimensions of biases tensor
 * \return Dimensions of biases tensor
 */
const services::Collection<size_t> Input::getBiasesSizes(const layers::Parameter *parameter) const
{
    return services::Collection<size_t>();
}


/**
 * Checks an input object for the 2D pooling layer
 * \param[in] parameter %Parameter of the layer
 * \param[in] method    Computation method of the layer
 */
void Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    layers::forward::Input::check(parameter, method);
    const Parameter *param = static_cast<const Parameter *>(parameter);
    data_management::TensorPtr dataTensor = get(layers::forward::data);
    const services::Collection<size_t> &dataDims = dataTensor->getDimensions();
    size_t nDim = dataDims.size();

    DAAL_CHECK_EX(dataTensor->getDimensions().size() == 4, services::ErrorIncorrectNumberOfDimensionsInTensor, services::ArgumentName, dataStr());

    for (size_t i = 0; i < 2; i++)
    {
        size_t spatialDimension = param->indices.size[i];
        if (spatialDimension > nDim - 1)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            error->setId(services::ErrorIncorrectParameter);
            error->addStringDetail(services::ArgumentName, "indices");
            this->_errors->add(error);
            return;
        }
    }
    if (param->indices.size[0] == param->indices.size[1])
    {
        services::SharedPtr<services::Error> error(new services::Error());
        error->setId(services::ErrorIncorrectParameter);
        error->addStringDetail(services::ArgumentName, "indices");
        this->_errors->add(error);
        return;
    }
}

/** Default constructor */
Result::Result() {}

/**
 * Returns dimensions of value tensor
 * \return Dimensions of value tensor
 */
const services::Collection<size_t> Result::getValueSize(const services::Collection<size_t> &inputSize,
        const daal::algorithms::Parameter *par, const int method) const
{
    services::Collection<size_t> valueDims = computeValueDimensions(inputSize, static_cast<const Parameter *>(par));
    return valueDims;
}

/**
 * Checks the result of the forward 2D pooling layer
 * \param[in] input %Input object for the layer
 * \param[in] parameter %Parameter of the layer
 * \param[in] method Computation method
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *param = static_cast<const Parameter *>(parameter);

    data_management::TensorPtr dataTensor = algInput->get(layers::forward::data);
    const services::Collection<size_t> &dataDims = dataTensor->getDimensions();

    DAAL_CHECK_EX(get(layers::forward::value)->getDimensions().size() == 2, services::ErrorIncorrectNumberOfDimensionsInTensor, services::ArgumentName, valueStr());

    services::Collection<size_t> valueDims = getValueSize(dataDims, param, method);

    if (!data_management::checkTensor(get(layers::forward::value).get(), this->_errors.get(), valueStr(), &valueDims)) { return; }

    services::SharedPtr<LayerData> layerData = get(layers::forward::resultForBackward);
    if (!layerData && param->predictionStage == false) { this->_errors->add(services::ErrorNullLayerData); return; }

    if (layerData && layerData->size() != 1)
        if (!layerData) { this->_errors->add(services::ErrorIncorrectSizeOfLayerData); return; }
}

services::Collection<size_t> Result::computeValueDimensions(const services::Collection<size_t> &inputDims, const Parameter *param)
{
    services::Collection<size_t> valueDims(2);
    valueDims[0] = inputDims[0];
    size_t pow4 = 1;
    for(size_t i = 0; i < param->pyramidHeight; i++)
    {
        pow4 *= 4;
    }
    valueDims[1] = inputDims[6 - param->indices.size[0] - param->indices.size[1]] * (pow4 - 1) / 3;
    return valueDims;
}

data_management::NumericTablePtr Result::createAuxInputDimensions(const services::Collection<size_t> &dataDims) const
{
    size_t nInputDims = dataDims.size();
    data_management::HomogenNumericTable<int> *auxInputDimsTable = new data_management::HomogenNumericTable<int>(
        nInputDims, 1, data_management::NumericTableIface::doAllocate);

    if (auxInputDimsTable == 0) { return data_management::NumericTablePtr(); }
    if (auxInputDimsTable->getArray() == 0) { return data_management::NumericTablePtr(); }

    int *auxInputDimsData = auxInputDimsTable->getArray();
    for (size_t i = 0; i < nInputDims; i++)
    {
        auxInputDimsData[i] = (int)dataDims[i];
    }
    return data_management::NumericTablePtr(auxInputDimsTable);
}

}// namespace interface1
}// namespace forward
}// namespace spatial_pooling2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
