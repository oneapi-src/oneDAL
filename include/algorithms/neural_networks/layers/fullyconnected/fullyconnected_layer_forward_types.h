/* file: fullyconnected_layer_forward_types.h */
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
//  Implementation of forward fully-connected layer.
//--
*/

#ifndef __FULLYCONNECTED_LAYER_FORWARD_TYPES_H__
#define __FULLYCONNECTED_LAYER_FORWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace fullyconnected
{
/**
 * @defgroup fullyconnected_forward Forward Fully-connected Layer
 * \copydoc daal::algorithms::neural_networks::layers::fullyconnected::forward
 * @ingroup fullyconnected
 * @{
 */
/**
 * \brief Contains classes for the forward fully-connected layer
 */
namespace forward
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FULLYCONNECTED__FORWARD__INPUT"></a>
 * \brief %Input objects for the forward fully-connected layer
 */
class Input : public layers::forward::Input
{
public:
    /**
     * Default constructor
     */
    Input() {};

    virtual ~Input() {}

    /**
     * Sets an input object for the forward fully-connected layer
     */
    using layers::forward::Input::set;
    /**
     * Returns an input object for the forward fully-connected layer
    */
    using layers::forward::Input::get;

    /**
    * Allocates memory to store the result of forward  fully-connected layer
     * \param[in] parameter %Parameter of forward fully-connected layer
     * \param[in] method    Computation method for the layer
    */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Parameter *parameter, const int method)
    {
        using daal::services::SharedPtr;
        using daal::data_management::Tensor;
        using daal::data_management::HomogenTensor;

        const Parameter *param =  static_cast<const Parameter *>(parameter);


        if( !get(layers::forward::weights) )
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(layers::forward::weights, getWeightsSizes(param));
        }

        if( !get(layers::forward::biases) )
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(layers::forward::biases, getBiasesSizes(param));
        }
    }

    /**
     * Returns dimensions of weights tensor
     * \return Dimensions of weights tensor
     */
    virtual const services::Collection<size_t> getWeightsSizes(const layers::Parameter *parameter) const DAAL_C11_OVERRIDE
    {
        using daal::services::Collection;

        const Parameter *param =  static_cast<const Parameter *>(parameter);

        Collection<size_t> wDims = get(layers::forward::data)->getDimensions();
        wDims[0] = param->nOutputs;

        return wDims;
    }

    /**
     * Returns dimensions of biases tensor
     * \return Dimensions of biases tensor
     */
    virtual const services::Collection<size_t> getBiasesSizes(const layers::Parameter *parameter) const DAAL_C11_OVERRIDE
    {
        using daal::services::Collection;

        const Parameter *param =  static_cast<const Parameter *>(parameter);
        size_t m = param->nOutputs;

        Collection<size_t> bDims;
        bDims.push_back( m );

        return bDims;
    }

    /**
     * Checks input object of the forward fully-connected layer
     * \param[in] parameter %Parameter of layer
     * \param[in] method    Computation method of the layer
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        layers::forward::Input::check(parameter, method);
        if( this->_errors->size() > 0 ) { return; }

        const Parameter *algParameter = static_cast<const Parameter *>(parameter);

        services::SharedPtr<data_management::Tensor> wTensor = get(layers::forward::weights);
        services::SharedPtr<data_management::Tensor> bTensor = get(layers::forward::biases);

        if( wTensor )
        {
        services::Collection<size_t> wDims = getWeightsSizes(algParameter);
        if (!data_management::checkTensor(wTensor.get(), this->_errors.get(), weightsStr(), &wDims)) { return; }
        }

        if( bTensor )
        {
        services::Collection<size_t> bDims = getBiasesSizes(algParameter);
        if (!data_management::checkTensor(bTensor.get(), this->_errors.get(), biasesStr(), &bDims)) { return; }
        }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FULLYCONNECTED__FORWARD__RESULT"></a>
 * \brief Results obtained with the compute() method of the forward fully-connected layer
 *        in the batch processing mode
 */
class Result : public layers::forward::Result
{
public:
    /**
     * Default constructor
     */
    Result() {}

    virtual ~Result() {}

    /**
     * Returns the result of the forward fully-connected layer
     */
    using layers::forward::Result::get;

    /**
     * Sets the result of the forward fully-connected layer
     */
    using layers::forward::Result::set;

    /**
     * Allocates memory to store the result of forward  fully-connected layer
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of forward fully-connected layer
     * \param[in] method    Computation method for the layer
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Input *in = static_cast<const Input * >(input);

        if (!get(layers::forward::value))
        {
            const services::Collection<size_t> &valueDims = getValueSize(in->get(layers::forward::data)->getDimensions(), parameter, method);
            DAAL_ALLOCATE_TENSOR_AND_SET(layers::forward::value, valueDims);
        }

        const layers::Parameter *par = static_cast<const layers::Parameter * >(parameter);
        if(!par->predictionStage)
        {
            if (!get(layers::forward::resultForBackward))
            {
                set(layers::forward::resultForBackward, services::SharedPtr<LayerData>(new LayerData()));
            }
            setResultForBackward(input);
        }
    }

    /**
     * Sets the result that is used in backward fully-connected layer
     * \param[in] input     Pointer to an object containing the input data
     */
    virtual void setResultForBackward(const daal::algorithms::Input *input) DAAL_C11_OVERRIDE
    {
        const Input *in = static_cast<const Input *>(input);
        set(auxData,    in->get(layers::forward::data));
        set(auxWeights, in->get(layers::forward::weights));
    }

    /**
     * Returns dimensions of value tensor
     * \return Dimensions of value tensor
     */
    virtual const services::Collection<size_t> getValueSize(const services::Collection<size_t> &inputSize,
                                                            const daal::algorithms::Parameter *par, const int method) const DAAL_C11_OVERRIDE
    {
        const Parameter *param =  static_cast<const Parameter *>(par);

        services::Collection<size_t> valueDims;
        valueDims.push_back(inputSize[0]);
        valueDims.push_back(param->nOutputs);
        return valueDims;
    }

    /**
     * Returns the result of forward fully-connected layer
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::Tensor> get(LayerDataId id) const
    {
        services::SharedPtr<layers::LayerData> layerData =
            services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
        if(!layerData)
        {
            this->_errors->add(services::ErrorNullLayerData);
            return services::SharedPtr<data_management::Tensor>();
        }
        return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
    }

    /**
     * Sets the result of forward fully-connected layer
     * \param[in] id     Identifier of the result
     * \param[in] value  Result
     */
    void set(LayerDataId id, const services::SharedPtr<data_management::Tensor> &value)
    {
        services::SharedPtr<layers::LayerData> layerData =
            services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
        if(!layerData)
        {
            this->_errors->add(services::ErrorNullLayerData);
        }
        (*layerData)[id] = value;
    }

    /**
     * Checks the result of the forward fully-connected layer
     * \param[in] input   %Input object of the layer
     * \param[in] par     %Parameter of the layer
     * \param[in] method  Computation method of the layer
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        layers::forward::Result::check(input, par, method);
        if( this->_errors->size() > 0 ) { return; }

        const Input     *algInput     = static_cast<const Input *>(input);
        const Parameter *algParameter = static_cast<const Parameter *>(par);

        services::SharedPtr<LayerData> layerData = get(layers::forward::resultForBackward);
        if (!layerData && algParameter->predictionStage == false) { this->_errors->add(services::ErrorNullLayerData); return; }

        services::SharedPtr<data_management::Tensor> dataTensor  = algInput->get(layers::forward::data);

        const services::Collection<size_t> &dataDims = dataTensor->getDimensions();
        const services::Collection<size_t>     wDims = algInput->getWeightsSizes(algParameter);
        const services::Collection<size_t>   valDims = getValueSize(dataDims, algParameter, defaultDense);

        if (!data_management::checkTensor(get(layers::forward::value).get(), this->_errors.get(), valueStr(), &valDims)) { return; }
        if(!algParameter->predictionStage)
        {
            if (!data_management::checkTensor(get(auxData).get(), this->_errors.get(), auxDataStr(), &dataDims)) { return; }
            if (!data_management::checkTensor(get(auxWeights).get(), this->_errors.get(), auxWeightsStr(), &wDims)) { return; }
        }
    }

    /**
     * Returns the serialization tag of the forward fully-connected layer result
     * \return     Serialization tag of the forward fully-connected layer result
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_FULLYCONNECTED_FORWARD_RESULT_ID; }

    /**
     * Serializes the object
     * \param[in]  arch  Storage for the serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
     * Deserializes the object
     * \param[in]  arch  Storage for the deserialized object or data structure
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
} // namespace fullyconnected
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
