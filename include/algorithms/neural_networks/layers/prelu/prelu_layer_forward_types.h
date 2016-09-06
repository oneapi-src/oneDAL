/* file: prelu_layer_forward_types.h */
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
//  Implementation of the parametric rectifier linear unit (prelu) layer
//--
*/

#ifndef __PRELU_LAYER_FORWARD_TYPES_H__
#define __PRELU_LAYER_FORWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"
#include "algorithms/neural_networks/layers/prelu/prelu_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * \brief Contains classes the prelu layer
 */
namespace prelu
{
/**
 * @defgroup prelu_forward Forward Parametric Rectifier Linear Unit (pReLU) Layer
 * \copydoc daal::algorithms::neural_networks::layers::prelu::forward
 * @ingroup prelu
 * @{
 */
/**
* \brief Contains classes for the forward prelu layer
*/
namespace forward
{
/**
* \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
*/
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__PRELU__FORWARD__INPUT"></a>
 * \brief %Input objects for the forward prelu layer
 */
class Input : public layers::forward::Input
{
public:
    /** \brief Default constructor */
    Input() {};

    /**
    * Gets the input of the forward prelu layer
    */
    using layers::forward::Input::get;

    /**
    * Sets the input of the forward prelu layer
    */
    using layers::forward::Input::set;

    virtual ~Input() {}

    /**
    * Allocates memory to store weights for the forward prelu layer
    * \param[in] par      %Parameter of the algorithm
    * \param[in] method   Computation method for the algorithm
    */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Parameter *par, const int method)
    {
        using daal::data_management::Tensor;
        using daal::data_management::HomogenTensor;

        const Parameter *parameter = static_cast<const Parameter * >(par);
        services::SharedPtr<Tensor> dataTensor = get(layers::forward::data);

        size_t wStartDim = parameter->dataDimension;
        size_t wDimNumber = parameter->weightsDimension;

        if(wStartDim > dataTensor->getNumberOfDimensions() - 1 || wStartDim + wDimNumber > dataTensor->getNumberOfDimensions())
        { this->_errors->add(services::ErrorIncorrectParameter); return; }

        if( get(layers::forward::weights) == 0 )
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(layers::forward::weights, getWeightsSizes(parameter));
        }
    }

    /**
    * Returns dimensions of weights tensor
    * \return Dimensions of weights tensor
    */
    virtual const services::Collection<size_t> getWeightsSizes(const layers::Parameter *par) const DAAL_C11_OVERRIDE
    {
        const Parameter *parameter =  static_cast<const Parameter *>(par);
        services::SharedPtr<data_management::Tensor> dataTensor = get(layers::forward::data);

        size_t wStartDim = parameter->dataDimension;
        size_t wDimNumber = parameter->weightsDimension;

        services::Collection<size_t> _dims = dataTensor->getDimensions();
        services::Collection<size_t> _wdims;

        for (size_t i = wStartDim; i < wStartDim + wDimNumber; i++)
        {
            _wdims.push_back(_dims[i]);
        }
        return _wdims;
    }

    /**
     * Returns dimensions of biases tensor
     * \return Dimensions of biases tensor
     */
    virtual const services::Collection<size_t> getBiasesSizes(const layers::Parameter *parameter) const DAAL_C11_OVERRIDE
    {
        return services::Collection<size_t>();
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__PRELU__FORWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method of the forward prelu layer
 */
class Result : public layers::forward::Result
{
public:
    /** \brief Default constructor */
    Result() : layers::forward::Result() {};
    virtual ~Result() {};

    /**
     * Returns the result of the forward prelu layer
     */
    using layers::forward::Result::get;

    /**
    * Sets the result of the forward prelu layer
    */
    using layers::forward::Result::set;

    /**
     * Returns the result of the forward prelu layer
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::Tensor> get(LayerDataId id) const
    {
        services::SharedPtr<layers::LayerData> layerData =
            services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
        return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
    }

    /**
     * Sets the result of the forward prelu layers
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(LayerDataId id, const services::SharedPtr<data_management::Tensor> &value)
    {
        services::SharedPtr<layers::LayerData> layerData =
            services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
        (*layerData)[id] = value;
    }

    /**
     * Checks the result of the forward prelu layer
     * \param[in] input   %Input object for the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        layers::forward::Result::check(input, par, method);
        const Input *in = static_cast<const Input * >(input);
        const Parameter *parameter =  static_cast<const Parameter *>(par);
        const services::Collection<size_t> weightsDimension = in->getWeightsSizes(parameter);

        if (!data_management::checkTensor(in->get(layers::forward::data).get(), this->_errors.get(), dataStr())) { return; }
        if (!data_management::checkTensor(in->get(layers::forward::weights).get(), this->_errors.get(), weightsStr(), &weightsDimension)) { return; }
        if (!data_management::checkTensor(get(layers::forward::value).get(), this->_errors.get(), valueStr())) { return; }
    }

    /**
     * Returns dimensions of value tensor
     * \return Dimensions of value tensor
     */
    virtual const services::Collection<size_t> getValueSize(const services::Collection<size_t> &inputSize,
                                                            const daal::algorithms::Parameter *par, const int method) const DAAL_C11_OVERRIDE
    {
        return inputSize;
    }

    /**
    * Allocates memory to store the result of the forward prelu layer
    * \param[in] input    Pointer to an object containing the input data
    * \param[in] par      %Parameter of the algorithm
    * \param[in] method   Computation method for the algorithm
    */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
    {
        const Input *in = static_cast<const Input *>(input);

        if (!get(layers::forward::value))
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(layers::forward::value, in->get(layers::forward::data)->getDimensions());
        }
        const layers::Parameter *parameter = static_cast<const layers::Parameter * >(par);
        if(!parameter->predictionStage)
        {
            if (!get(layers::forward::resultForBackward))
            {
                set(layers::forward::resultForBackward, services::SharedPtr<LayerData>(new LayerData()));
            }
            setResultForBackward(input);
        }
    }

    /**
     * Sets the result that is used in backward prelu layer
     * \param[in] input     Pointer to an object containing the input data
     */
    virtual void setResultForBackward(const daal::algorithms::Input *input) DAAL_C11_OVERRIDE
    {
        const Input *in = static_cast<const Input *>(input);
        set(auxData, in->get(layers::forward::data));
        set(auxWeights, in->get(layers::forward::weights));
    }

    /**
     * Returns the serialization tag of the result
     * \return         Serialization tag of the result
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_PRELU_FORWARD_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for the serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive   *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes the object
    *  \param[in]  arch  Storage for the deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
} // namespace interface1
using interface1::Input;
using interface1::Result;
} // namespace forward
/** @} */
} // namespace prelu
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
