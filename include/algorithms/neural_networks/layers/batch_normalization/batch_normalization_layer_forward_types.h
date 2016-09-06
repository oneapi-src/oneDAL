/* file: batch_normalization_layer_forward_types.h */
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
//  Implementation of the forward batch normalization layer.
//--
*/

#ifndef __BATCH_NORMALIZATION_LAYER_FORWARD_TYPES_H__
#define __BATCH_NORMALIZATION_LAYER_FORWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"
#include "algorithms/neural_networks/layers/batch_normalization/batch_normalization_layer_types.h"

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
/**
 * @defgroup batch_normalization_forward Forward Batch Normalization Layer
 * \copydoc daal::algorithms::neural_networks::layers::batch_normalization::forward
 * @ingroup batch_normalization
 * @{
 */
/**
 * \brief Contains classes for forward batch normalization layer
 */
namespace forward
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCH_NORMALIZATION__FORWARD__INPUTLAYERDATAID"></a>
 * Available identifiers of input objects for the forward batch normalization layer
 */
enum InputLayerDataId
{
    populationMean     = 3, /*!< 1-dimensional tensor of size \f$n_k\f$ that stores population mean computed on the previous stage */
    populationVariance = 4  /*!< 1-dimensional tensor of size \f$n_k\f$ that stores population variance computed on the previous stage */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCH_NORMALIZATION__FORWARD__INPUT"></a>
 * \brief %Input objects for the forward batch normalization layer.
 */
class Input : public layers::forward::Input
{
public:
    /** Default constructor */
    Input() : layers::forward::Input(5) {}

    virtual ~Input() {}

    /**
     * Returns an input object for the forward batch normalization layer
     */
    using layers::forward::Input::get;
    /**
    * Sets an input object for the forward batch normalization layer
    */
    using layers::forward::Input::set;


    /**
     * Allocates memory to store input objects of forward batch normalization layer
     * \param[in] parameter %Parameter of forward batch normalization layer
     * \param[in] method    Computation method for the layer
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Parameter *parameter, const int method)
    {
        const Parameter *param =  static_cast<const Parameter *>(parameter);

        if (!get(layers::forward::weights))
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(layers::forward::weights, getWeightsSizes(param));
        }

        if (!get(layers::forward::biases))
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
        const Parameter *algParameter =  static_cast<const Parameter *>(parameter);
        const services::Collection<size_t> &dims = get(layers::forward::data)->getDimensions();
        services::Collection<size_t> wDims(1);
        wDims[0] = dims[algParameter->dimension];
        return wDims;
    }

    /**
     * Returns dimensions of biases tensor
     * \return Dimensions of biases tensor
     */
    virtual const services::Collection<size_t> getBiasesSizes(const layers::Parameter *parameter) const DAAL_C11_OVERRIDE
    {
        return getWeightsSizes(parameter);
    }

    /**
     * Returns an input object for forward batch normalization layer
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::Tensor> get(InputLayerDataId id) const
    {
        return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets input for the forward batch normalization layer
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Input object to set
     */
    void set(InputLayerDataId id, const services::SharedPtr<data_management::Tensor> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks input object of the forward batch normalization layer
     * \param[in] parameter %Parameter of layer
     * \param[in] method    Computation method of the layer
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        size_t dimension = algParameter->dimension;
        services::SharedPtr<data_management::Tensor> dataTensor = get(layers::forward::data);

        if (!data_management::checkTensor(dataTensor.get(), this->_errors.get(), dataStr())) { return; }

        size_t dimensionSize = dataTensor->getDimensionSize(dimension);
        services::Collection<size_t> weightDims(1);
        weightDims[0] = dimensionSize;

        if (get(layers::forward::weights))
        {
            if (!data_management::checkTensor(get(layers::forward::weights).get(), this->_errors.get(), weightsStr(), &weightDims)) { return; }
        }
        if (get(layers::forward::biases))
        {
            if (!data_management::checkTensor(get(layers::forward::biases).get(),  this->_errors.get(), biasesStr(), &weightDims)) { return; }
        }
        if (!data_management::checkTensor(get(populationMean).get(),           this->_errors.get(), populationMeanStr(), &weightDims)) { return; }
        if (!data_management::checkTensor(get(populationVariance).get(),       this->_errors.get(), populationVarianceStr(), &weightDims)) { return; }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCH_NORMALIZATION__FORWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the forward batch normalization layer
 */
class Result : public layers::forward::Result
{
public:
    /** Default Constructor */
    Result() {}
    virtual ~Result() {}

    using layers::forward::Result::get;
    using layers::forward::Result::set;

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
     * Allocates memory to store the result of the forward batch normalization layer
     * \param[in] input Pointer to an object containing the input data
     * \param[in] parameter %Parameter of the forward batch normalization layer
     * \param[in] method Computation method for the layer
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Input *in = static_cast<const Input *>(input);
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);

        if (!get(layers::forward::value))
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(layers::forward::value, in->get(layers::forward::data)->getDimensions());
        }
        if (!get(layers::forward::resultForBackward))
        {
            set(layers::forward::resultForBackward, services::SharedPtr<LayerData>(new LayerData()));
        }

        size_t dimension = algParameter->dimension;
        size_t dimensionSize = in->get(layers::forward::data)->getDimensionSize(dimension);
        services::Collection<size_t> auxDims(1);
        auxDims[0] = dimensionSize;
        if (!get(auxMean))
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(auxMean, auxDims);
        }
        if (!get(auxStandardDeviation))
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(auxStandardDeviation, auxDims);
        }

        const layers::Parameter *par = static_cast<const layers::Parameter * >(parameter);
        if(!par->predictionStage)
        {
            setResultForBackward(input);
            if (!get(auxPopulationMean))
            {
                DAAL_ALLOCATE_TENSOR_AND_SET(auxPopulationMean, auxDims);
            }
            if (!get(auxPopulationVariance))
            {
                DAAL_ALLOCATE_TENSOR_AND_SET(auxPopulationVariance, auxDims);
            }
        }
    }

    /**
     * Sets the result that is used in backward batch normalization layer
     * \param[in] input     Pointer to an object containing the input data
     */
    virtual void setResultForBackward(const daal::algorithms::Input *input) DAAL_C11_OVERRIDE
    {
        const Input *in = static_cast<const Input *>(input);
        set(auxData,    in->get(layers::forward::data));
        set(auxWeights, in->get(layers::forward::weights));
    }

    /**
     * Returns the result of the forward batch normalization layer
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::Tensor> get(LayerDataId id) const
    {
        services::SharedPtr<layers::LayerData> layerData = get(layers::forward::resultForBackward);
        return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
    }

    /**
     * Sets the result of the forward batch normalization layer
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Result
     */
    void set(LayerDataId id, const services::SharedPtr<data_management::Tensor> &ptr)
    {
        services::SharedPtr<layers::LayerData> layerData = get(layers::forward::resultForBackward);
        (*layerData)[id] = ptr;
    }

    /**
     * Checks the result of the forward batch normalization layer
     * \param[in] input     %Input of the layer
     * \param[in] parameter %Parameter of the layer
     * \param[in] method    Computation method of the layer
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        const Input *algInput = static_cast<const Input *>(input);
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        size_t dimension = algParameter->dimension;

        services::SharedPtr<data_management::Tensor> dataTensor = algInput->get(layers::forward::data);
        const services::Collection<size_t> &dataDims = dataTensor->getDimensions();

        services::SharedPtr<data_management::Tensor> valueTensor = get(layers::forward::value);
        if (!data_management::checkTensor(valueTensor.get(), this->_errors.get(), valueStr(), &dataDims)) { return; }

        size_t dimensionSize = valueTensor->getDimensionSize(dimension);
        services::Collection<size_t> auxDims(1);
        auxDims[0] = dimensionSize;

        services::SharedPtr<LayerData> layerData = get(layers::forward::resultForBackward);
        if (!layerData) { this->_errors->add(services::ErrorNullLayerData); return; }

        if (algParameter->predictionStage == false && layerData->size() != 6)
            if (!layerData) { this->_errors->add(services::ErrorIncorrectSizeOfLayerData); return; }

        if (algParameter->predictionStage == true && layerData->size() != 2)
            if (!layerData) { this->_errors->add(services::ErrorIncorrectSizeOfLayerData); return; }

        if (!data_management::checkTensor(get(auxMean).get(),               this->_errors.get(), auxMeanStr(), &auxDims)) { return; }
        if (!data_management::checkTensor(get(auxStandardDeviation).get(),  this->_errors.get(), auxStandardDeviationStr(), &auxDims)) { return; }
        if(!algParameter->predictionStage)
        {
            if (!data_management::checkTensor(get(auxData).get(),               this->_errors.get(), auxDataStr(), &dataDims)) { return; }
            if (!data_management::checkTensor(get(auxWeights).get(),            this->_errors.get(), auxWeightsStr(), &auxDims)) { return; }
            if (!data_management::checkTensor(get(auxPopulationMean).get(),     this->_errors.get(), auxPopulationMeanStr(), &auxDims)) { return; }
            if (!data_management::checkTensor(get(auxPopulationVariance).get(), this->_errors.get(), auxPopulationVarianceStr(), &auxDims)) { return; }
        }
    }

    /**
     * Returns the serialization tag of the forward batch normalization layer result
     * \return     Serialization tag of the forward batch normalization layer result
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_BATCH_NORMALIZATION_FORWARD_RESULT_ID; }

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
} // namespace batch_normalization
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif
