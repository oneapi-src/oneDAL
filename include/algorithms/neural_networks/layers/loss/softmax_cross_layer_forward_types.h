/* file: softmax_cross_layer_forward_types.h */
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
//  Implementation of the forward softmax cross-entropy layer types.
//--
*/

#ifndef __NEURAL_NENTWORK_LOSS_SOFTMAX_CROSS_LAYER_FORWARD_TYPES_H__
#define __NEURAL_NENTWORK_LOSS_SOFTMAX_CROSS_LAYER_FORWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"
#include "algorithms/neural_networks/layers/loss/loss_layer_forward_types.h"
#include "algorithms/neural_networks/layers/loss/softmax_cross_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace loss
{
namespace softmax_cross
{
/**
 * @defgroup softmax_cross_forward Forward Softmax Cross-entropy Layer
 * \copydoc daal::algorithms::neural_networks::layers::loss::softmax_cross::forward
 * @ingroup softmax_cross
 * @{
 */
/**
 * \brief Contains classes for the forward softmax cross-entropy layer
 */
namespace forward
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__SOFTMAX_CROSS__FORWARD__INPUT"></a>
 * \brief %Input objects for the forward softmax cross-entropy layer
 */
class Input : public loss::forward::Input
{
public:
    /** Default constructor */
    Input() : loss::forward::Input() {};

    /**
     * Returns an input object for the forward softmax cross-entropy layer
     */
    using loss::forward::Input::get;

    /**
     * Sets an input object for the forward softmax cross-entropy layer
     */
    using loss::forward::Input::set;

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

    /**
     * Checks an input object for the layer algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method of the algorithm
     */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 5) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        if (!data_management::checkTensor(get(layers::forward::data).get(), this->_errors.get(), dataStr())) { return; }
    }

    virtual ~Input() {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__SOFTMAX_CROSS__FORWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the forward softmax cross-entropy layer
 */
class Result : public loss::forward::Result
{
public:
    /** Default constructor */
    Result() : loss::forward::Result() {};
    virtual ~Result() {};

    /**
     * Returns the result of the forward softmax cross-entropy layer
     */
    using loss::forward::Result::get;

    /**
     * Sets the result of the forward softmax cross-entropy layer
     */
    using loss::forward::Result::set;

    /**
     * Returns the result of the forward softmax cross-entropy layer
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
     * Sets the result of the forward softmax cross-entropy layer
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
     * Checks the result of the forward softmax cross-entropy layer
     * \param[in] input   %Input object for the layer
     * \param[in] par     %Parameter of the layer
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        const Input *in = static_cast<const Input * >(input);
        layers::forward::Result::check(input, par, method);
        services::Collection<size_t> valueDim(1);
        valueDim[0] = 1;
        if (!data_management::checkTensor(get(layers::forward::value).get(), this->_errors.get(), valueStr(), &valueDim)) { return; }
        if (!data_management::checkTensor(get(auxProbabilities).get(), this->_errors.get(), auxProbabilitiesStr(), &(in->get(layers::forward::data)->getDimensions()))) { return; }
        const layers::Parameter *parameter = static_cast<const layers::Parameter * >(par);
        if(!parameter->predictionStage)
        {
            if (!data_management::checkTensor(get(auxGroundTruth).get(), this->_errors.get(), auxGroundTruthStr(), &(in->get(loss::forward::groundTruth)->getDimensions()))) { return; }
        }
    }

    /**
     * Allocates memory to store the result of the forward softmax cross-entropy layer
     * \param[in] input Pointer to an object containing the input data
     * \param[in] method Computation method for the layer
     * \param[in] parameter %Parameter of the forward softmax cross-entropy layer
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Input *in = static_cast<const Input * >(input);

        services::Collection<size_t> valueDim(1);
        valueDim[0] = 1;
        if (!get(layers::forward::value))
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(layers::forward::value, valueDim);
        }

        if (!get(layers::forward::resultForBackward))
        {
            set(layers::forward::resultForBackward, services::SharedPtr<LayerData>(new LayerData()));
        }
        if (!get(auxProbabilities))
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(auxProbabilities, in->get(layers::forward::data)->getDimensions());
        }

        const layers::Parameter *par = static_cast<const layers::Parameter * >(parameter);
        if(!par->predictionStage)
        {
            setResultForBackward(input);
        }
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
    * Returns the serialization tag of the forward softmax cross-entropy layer result
    * \return         Serialization tag of the forward softmax cross-entropy layer result
    */
    int getSerializationTag() DAAL_C11_OVERRIDE { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_LOSS_SOFTMAX_CROSS_FORWARD_RESULT_ID; }

    /**
     * Sets the result that is used in backward abs layer
     * \param[in] input     Pointer to an object containing the input data
     */
    virtual void setResultForBackward(const daal::algorithms::Input *input) DAAL_C11_OVERRIDE
    {
        const loss::forward::Input *in = static_cast<const loss::forward::Input * >(input);
        set(auxGroundTruth, in->get(loss::forward::groundTruth));
    }

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
} // namespace softmax_cross
} // namespace loss
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
