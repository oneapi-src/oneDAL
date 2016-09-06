/* file: batch_normalization_layer_backward_types.h */
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
//  Implementation of the backward batch normalization layer.
//--
*/

#ifndef __BATCH_NORMALIZATION_LAYER_BACKWARD_TYPES_H__
#define __BATCH_NORMALIZATION_LAYER_BACKWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_backward_types.h"
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
 * @defgroup batch_normalization_backward Backward Batch Normalization Layer
 * \copydoc daal::algorithms::neural_networks::layers::batch_normalization::backward
 * @ingroup batch_normalization
 * @{
 */
/**
 * \brief Contains classes for the backward batch normalization layer
 */
namespace backward
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCH_NORMALIZATION__BACKWARD__INPUT"></a>
 * \brief %Input objects for the backward batch normalization layer
 */
class Input : public layers::backward::Input
{
public:
    /** Default constructor */
    Input() {};

    virtual ~Input() {}

    /**
     * Returns an input object for the backward batch normalization layer
     */
    using layers::backward::Input::get;

    /**
     * Sets an input object for the backward batch normalization layer
     */
    using layers::backward::Input::set;

    /**
     * Returns an input object for the backward batch normalization layer
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::Tensor> get(LayerDataId id) const
    {
        services::SharedPtr<layers::LayerData> inputData = get(layers::backward::inputFromForward);
        return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*inputData)[id]);
    }

    /**
     * Sets an input object for the backward batch normalization layer
     * \param[in] id  Identifier of the input object
     * \param[in] ptr Pointer to the object
     */
    void set(LayerDataId id, const services::SharedPtr<data_management::Tensor> &ptr)
    {
        services::SharedPtr<layers::LayerData> inputData = get(layers::backward::inputFromForward);
        (*inputData)[id] = ptr;
    }

    /**
     * Checks an input object for the backward batch normalization layer
     * \param[in] parameter Layer parameter
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        size_t dimension = algParameter->dimension;

        services::SharedPtr<data_management::Tensor> inputGradientTensor = get(layers::backward::inputGradient);
        if (!data_management::checkTensor(inputGradientTensor.get(), this->_errors.get(), inputGradientStr())) { return; }

        const services::Collection<size_t> &dataDims = inputGradientTensor->getDimensions();

        size_t dimensionSize = dataDims[dimension];
        services::Collection<size_t> auxDims(1);
        auxDims[0] = dimensionSize;

        if (!data_management::checkTensor(get(auxData).get(),               this->_errors.get(), auxDataStr(),              &dataDims)) { return; }
        if (!data_management::checkTensor(get(auxWeights).get(),            this->_errors.get(), auxWeightsStr(),            &auxDims)) { return; }
        if (!data_management::checkTensor(get(auxMean).get(),               this->_errors.get(), auxMeanStr(),               &auxDims)) { return; }
        if (!data_management::checkTensor(get(auxStandardDeviation).get(),  this->_errors.get(), auxStandardDeviationStr(),  &auxDims)) { return; }
        if (!data_management::checkTensor(get(auxPopulationMean).get(),     this->_errors.get(), auxPopulationMeanStr(),     &auxDims)) { return; }
        if (!data_management::checkTensor(get(auxPopulationVariance).get(), this->_errors.get(), auxPopulationVarianceStr(), &auxDims)) { return; }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCH_NORMALIZATION__BACKWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the backward batch normalization layer
 */
class Result : public layers::backward::Result
{
public:
    /** Default constructor */
    Result() {}
    virtual ~Result() {}

    /**
     * Returns an result object for the backward batch normalization layer
     */
    using layers::backward::Result::get;

    /**
     * Sets an result object for the backward batch normalization layer
     */
    using layers::backward::Result::set;

    /**
     * Allocates memory to store the result of the backward batch normalization layer
     * \param[in] input Pointer to an object containing the input data
     * \param[in] parameter %Parameter of the backward batch normalization layer
     * \param[in] method Computation method for the layer
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Input *in = static_cast<const Input *>(input);
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);

        services::SharedPtr<data_management::Tensor> inputGradientTensor = in->get(layers::backward::inputGradient);

        if (!data_management::checkTensor(inputGradientTensor.get(), this->_errors.get(), inputGradientStr())) { return; }

        DAAL_ALLOCATE_TENSOR_AND_SET(layers::backward::gradient, inputGradientTensor->getDimensions());

        size_t dimension = algParameter->dimension;
        size_t dimensionSize = inputGradientTensor->getDimensionSize(dimension);
        services::Collection<size_t> requiredDimensionSizes(1);
        requiredDimensionSizes[0] = dimensionSize;

        if (!get(layers::backward::weightDerivatives))
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(layers::backward::weightDerivatives, requiredDimensionSizes);
        }
        if (!get(layers::backward::biasDerivatives))
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(layers::backward::biasDerivatives, requiredDimensionSizes);
        }
    }

    /**
     * Checks the result of the backward batch normalization layer
     * \param[in] input     %Input object for the layer
     * \param[in] parameter %Parameter of the layer
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        const Input *algInput = static_cast<const Input *>(input);
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        size_t dimension = algParameter->dimension;
        services::SharedPtr<data_management::Tensor> gradientTensor = get(layers::backward::gradient);

        services::SharedPtr<data_management::Tensor> inputGradientTensor = algInput->get(layers::backward::inputGradient);
        const services::Collection<size_t> &gradientDims = inputGradientTensor->getDimensions();

        if (!data_management::checkTensor(gradientTensor.get(), this->_errors.get(), gradientStr(), &gradientDims)) { return; }

        size_t dimensionSize = gradientTensor->getDimensionSize(dimension);
        services::Collection<size_t> derDims(1);
        derDims[0] = dimensionSize;

        if (!data_management::checkTensor(get(layers::backward::weightDerivatives).get(), this->_errors.get(), weightDerivativesStr(), &derDims)) { return; }
        if (!data_management::checkTensor(get(layers::backward::biasDerivatives).get(), this->_errors.get(), biasDerivativesStr(), &derDims)) { return; }
    }

    /**
     * Returns the serialization tag of the backward batch normalization layer result
     * \return     Serialization tag of the backward batch normalization layer result
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_BATCH_NORMALIZATION_BACKWARD_RESULT_ID; }

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
} // namespace backward

/** @} */
} // namespace batch_normalization
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif
