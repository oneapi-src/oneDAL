/* file: softmax_layer_forward_types.h */
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
//  Implementation of the forward softmax layer interface.
//--
*/

#ifndef __SOFTMAX_LAYER_FORWARD_TYPES_H__
#define __SOFTMAX_LAYER_FORWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"
#include "algorithms/neural_networks/layers/softmax/softmax_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace softmax
{
/**
 * @defgroup softmax_layers_forward Forward Softmax Layer
 * \copydoc daal::algorithms::neural_networks::layers::softmax::forward
 * @ingroup softmax_layers
 * @{
 */
/**
 * \brief Contains classes of the forward softmax layer
 */
namespace forward
{
/**
* \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
*/
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SOFTMAX__FORWARD__INPUT"></a>
 * \brief %Input objects for the forward softmax layer
 */
class Input : public layers::forward::Input
{
public:
    /** Default constructor */
    Input() {};

    /**
     * Returns an input object for the forward softmax layer
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    using layers::forward::Input::get;

    /**
     * Sets an input object for the forward softmax layer
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    using layers::forward::Input::set;

    /**
     * Returns dimensions of weights tensor
     * \return Dimensions of weights tensor
     */
    virtual const services::Collection<size_t> getWeightsSizes(const layers::Parameter *parameter) const DAAL_C11_OVERRIDE
    {
        return services::Collection<size_t>();
    }

    /**
     * Returns dimensions of biases tensor
     * \return Dimensions of biases tensor
     */
    virtual const services::Collection<size_t> getBiasesSizes(const layers::Parameter *parameter) const DAAL_C11_OVERRIDE
    {
        return services::Collection<size_t>();
    }

    virtual ~Input() {}

    /**
     * Checks input object of the forward softmax layer
     * \param[in] par     Layer parameter
     * \param[in] method  Computation method of the layer
     */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        layers::forward::Input::check(par, method);

        services::SharedPtr<data_management::Tensor> dataTensor = get(layers::forward::data);
        const softmax::Parameter *parameter = static_cast<const softmax::Parameter *>(par);
        if (parameter->dimension >= dataTensor->getDimensions().size())
        {
            this->_errors->add(services::ErrorIncorrectParameter); return;
        }
    }

};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SOFTMAX__FORWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the forward softmax layer
 */
class Result : public layers::forward::Result
{
public:
    /** Default constructor */
    Result() : layers::forward::Result() {};
    virtual ~Result() {};

    /**
     * Returns result of the forward softmax layer
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    using layers::forward::Result::get;

    /**
     * Sets the result of the forward softmax layer
     * \param[in] id      Identifier of the result
     * \param[in] value   Pointer to the object
     */
    using layers::forward::Result::set;

    /**
     * Returns result of the forward softmax layer
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::Tensor> get(LayerDataId id) const
    {
        services::SharedPtr<layers::LayerData> layerData =
            services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
        return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
    }

    /**
     * Sets the result of the forward softmax layer
     * \param[in] id      Identifier of the result
     * \param[in] value   Pointer to the object
     */
    void set(LayerDataId id, const services::SharedPtr<data_management::Tensor> &value)
    {
        services::SharedPtr<layers::LayerData> layerData =
            services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
        (*layerData)[id] = value;
    }

    /**
     * Checks the result of the forward softmax layer
     * \param[in] input   %Input object for the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        layers::forward::Result::check(input, par, method);

        const Input *in = static_cast<const Input *>(input);
        if (!data_management::checkTensor(in->get(layers::forward::data).get(), this->_errors.get(), dataStr())) { return; }
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
    * Allocates memory to store the result of the forward softmax layer
    * \param[in] input     Pointer to an object containing the input data
    * \param[in] par      %Parameter of the backward softmax layer
    * \param[in] method    Computation method
    */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
    {
        const layers::forward::Input *in = static_cast<const layers::forward::Input * >(input);

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
     * Sets the result that is used in backward softmax layer
     * \param[in] input     Pointer to an object containing the input data
     */
    virtual void setResultForBackward(const daal::algorithms::Input *input) DAAL_C11_OVERRIDE
    {
        const layers::forward::Input *in = static_cast<const layers::forward::Input * >(input);
        set(auxValue, get(layers::forward::value));
    }


    /**
     * Returns the serialization tag of the forward softmax layer result
     * \return         Serialization tag of the the forward softmax layer result
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_SOFTMAX_FORWARD_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for the serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
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
} // namespace softmax
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
