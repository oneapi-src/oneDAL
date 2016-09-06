/* file: split_layer_backward_types.h */
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
//  Implementation of the backward split layer
//--
*/

#ifndef __SPLIT_LAYER_BACKWARD_TYPES_H__
#define __SPLIT_LAYER_BACKWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_backward_types.h"
#include "algorithms/neural_networks/layers/split/split_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * \brief Contains classes for the split layer
 */
namespace split
{
/**
 * @defgroup split_backward Backward Split Layer
 * \copydoc daal::algorithms::neural_networks::layers::split::backward
 * @ingroup split
 * @{
 */
/**
 * \brief Contains classes for the backward split layer
 */
namespace backward
{
/**
* <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPLIT__BACKWARD__INPUTLAYERDATAID"></a>
* Available identifiers of input objects for the backward split layer
*/
enum InputLayerDataId
{
    inputGradientCollection = 1   /*!< Input structure retrieved from the result of the forward split layer */
};

/**
* \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
*/
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPLIT__BACKWARD__INPUT"></a>
 * \brief %Input parameters for the backward split layer
 */
class Input : public layers::backward::Input
{
public:
    /** \brief Default constructor */
    Input()
    {
        set(inputGradientCollection, services::SharedPtr<LayerData>(new LayerData()));
    }

    virtual ~Input() {}

    /**
     * Returns an input object for the backward split layer
     */
    using layers::backward::Input::get;

    /**
     * Sets an input object for the backward split layer
     */
    using layers::backward::Input::set;

    /**
     * Returns a tensor with a given index from the collection of input tensors
     * \param[in] id    Identifier of the collection of input tensors
     * \param[in] index Index of the tensor to be returned
     * \return          Pointer to the table with the input tensor
     */
    services::SharedPtr<data_management::Tensor> get(InputLayerDataId id, size_t index) const
    {
        services::SharedPtr<layers::LayerData> layerData = get(id);
        return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[index]);
    }

    /**
     * Returns input Tensor of the layer algorithm
     * \param[in] id    Identifier of the input tensor
     * \return          %Input tensor that corresponds to the given identifier
     */
    services::SharedPtr<LayerData> get(InputLayerDataId id) const
    {
        return services::staticPointerCast<LayerData, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets an input object for the backward split layer
     * \param[in] id     Identifier of the input object
     * \param[in] value  Pointer to the input object
     * \param[in] index  Index of the tensor to be set
     */
    void set(InputLayerDataId id, const services::SharedPtr<data_management::Tensor> &value, size_t index)
    {
        services::SharedPtr<layers::LayerData> layerData = get(id);
        (*layerData)[index] = value;
    }

    /**
    * Sets input for the layer algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(InputLayerDataId id, const services::SharedPtr<LayerData> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Adds tensor with input gradient to the input object of the backward split layer
     * \param[in] igTensor  Tensor with input gradient
     * \param[in] index     Index of the tensor with input gradient
     */
    virtual void addInputGradient(const services::SharedPtr<data_management::Tensor> &igTensor, size_t index) DAAL_C11_OVERRIDE
    {
        services::SharedPtr<LayerData> layerData = get(inputGradientCollection);

        size_t nInputs = layerData->size();
        (*(layerData))[nInputs] = igTensor;
        set(inputGradientCollection, layerData);
    }

    /**
     * Sets input structure retrieved from the result of the forward layer
     * \param[in] result Result of the forward layer
     */
    virtual void setInputFromForward(services::SharedPtr<layers::forward::Result> result) DAAL_C11_OVERRIDE
    {}

    /**
     * Checks an input object of the backward split layer
     * \param[in] par     Algorithm parameter
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        const Parameter *parameter = static_cast<const Parameter *>(par);

        if(Argument::size() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }
        services::SharedPtr<LayerData> layerData = get(inputGradientCollection);
        size_t nInputs = parameter->nInputs;

        if (layerData->size() != nInputs) {this->_errors->add(services::ErrorIncorrectSizeOfLayerData); return; };
        services::SharedPtr<data_management::Tensor> inputTensor0 = get(inputGradientCollection, 0);

        if (!data_management::checkTensor(inputTensor0.get(), this->_errors.get(), inputGradientCollectionStr()))
        {
            size_t inx = this->_errors->size();
            (*this->_errors->getErrors() )[inx - 1]->addIntDetail(services::ArgumentName, (int)0);
            return;
        }

        services::Collection<size_t> dims0 = inputTensor0->getDimensions();

        for (size_t i = 1; i < nInputs; i++)
        {
            if (!data_management::checkTensor(get(inputGradientCollection, i).get(), this->_errors.get(), inputGradientCollectionStr(), &dims0))
            {
                size_t inx = this->_errors->size();
                (*this->_errors->getErrors() )[inx - 1]->addIntDetail(services::ArgumentName, (int)i);
                return;
            }
        }
    }

    /**
    * Returns the layout of the input object for the layer algorithm
    * \return Layout of the input object for the layer algorithm
    */
    virtual LayerInputLayout getLayout() const DAAL_C11_OVERRIDE { return collectionInput; }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPLIT__BACKWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method of the backward split layer
 */
class Result : public layers::backward::Result
{
public:
    /** \brief Default constructor */
    Result() : layers::backward::Result() {};
    virtual ~Result() {};

    /**
     * Returns the result of the backward split layer
     */
    using layers::backward::Result::get;

    /**
     * Sets the result of the backward split layer
     */
    using layers::backward::Result::set;

    /**
     * Checks the result of the backward split layer
     * \param[in] input   %Input object for the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        const Input *algInput = static_cast<const Input *>(input);

        if(Argument::size() != 4) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        services::SharedPtr<data_management::Tensor> inputTensor = algInput->get(inputGradientCollection, 0);
        services::Collection<size_t> dims = inputTensor->getDimensions();

        if (!data_management::checkTensor(get(layers::backward::gradient).get(), this->_errors.get(), gradientStr(), &dims)) { return; }
    }

    /**
    * Allocates memory to store the result of the backward split layer
     * \param[in] input     Pointer to an object containing the input data
     * \param[in] method    Computation method for the algorithm
     * \param[in] parameter %Parameter of the backward split layer
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        if (!get(layers::backward::gradient))
        {
            const Input *in = static_cast<const Input *>(input);

            services::SharedPtr<data_management::Tensor> valueTable = in->get(inputGradientCollection, 0);

            if(!valueTable) { this->_errors->add(services::ErrorNullInputNumericTable); return; }

            if (!get(layers::backward::gradient))
            {
                DAAL_ALLOCATE_TENSOR_AND_SET(layers::backward::gradient, valueTable->getDimensions());
            }
        }
    }

    /**
    * Returns the serialization tag of the result
    * \return     Serialization tag of the result
    */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_SPLIT_BACKWARD_RESULT_ID; }

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
} // namespace backward
/** @} */
} // namespace split
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
