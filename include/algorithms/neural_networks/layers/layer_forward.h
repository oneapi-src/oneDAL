/* file: layer_forward.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of neural network layer.
//--
*/

#ifndef __LAYER_FORWARD_H__
#define __LAYER_FORWARD_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for training and prediction using neural network
 */
namespace neural_networks
{
/**
 * \brief Contains classes for neural network layers
 */
namespace layers
{
/**
 * \brief Contains classes for the forward stage of the neural network layer
 */
namespace forward
{
namespace interface1
{
/**
 * @ingroup layers_forward
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARD__LAYERIFACE"></a>
 *  \brief Abstract class which defines interface for the layer
 */
class LayerIface : public daal::algorithms::Analysis<batch>
{
public:
    virtual ~LayerIface() {};

    /**
     * Returns the structure that contains results of the layer
     * \return Structure that contains results of the layer
     */
    virtual forward::ResultPtr getLayerResult() = 0;

    /**
     * Returns the structure that contains input objects of the layer
     * \return Structure that contains input objects of the layer
     */
    virtual Input *getLayerInput() = 0;

    /**
     * Returns the structure that contains parameters of the layer
     * \return Structure that contains parameters of the layer
     */
    virtual Parameter *getLayerParameter() = 0;

    /**
     * Returns a pointer to the newly allocated forward neural network layer with a copy of input objects
     * and parameters of this layer
     * \return Pointer to the newly allocated forward layer
     */
    daal::algorithms::neural_networks::layers::forward::LayerIfacePtr clone() const
    {
        return LayerIfacePtr(cloneImpl());
    }

    /**
     * Allocates memory buffers needed for the computations
     */
    virtual services::Status allocateResult() = 0;

    /**
     * Allocates memory buffers needed for the computations
     */
    virtual services::Status allocateInput() = 0;

    /**
     * Initializes values of weights and biases if needed
     */
    virtual services::Status initializeInput() = 0;
    /**
     * Connects two layers in neural network by getting tensor with value
     * from the result of the previous layer and adding it to the input object of this layer algorithm
     * \param[in] result        Structure that contains results of the previous layer
     * \param[in] resultIndex   Index of the tensor with value in the structure that contains
     *                          results of the previous layer
     * \param[in] inputIndex    Index in the input object of this layer algorithm
     *                          where the tensor with value should be placed
     */
    virtual services::Status addInput(forward::ResultPtr result, size_t resultIndex, size_t inputIndex) = 0;

protected:
    virtual LayerIface *cloneImpl() const = 0;
};

typedef services::SharedPtr<LayerIface> LayerIfacePtr;
/** @} */
/**
 * @ingroup layers_forward
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARD__LAYERIFACEIMPL"></a>
 *  \brief Implements the abstract interface LayerIface. LayerIfaceImpl is, in turn, the base class
 *         for the classes interfacing the layers.
 */
class LayerIfaceImpl : public LayerIface
{
public:
    virtual ~LayerIfaceImpl() {};

    /**
     * Allocates memory buffers needed for the computations
     */
    virtual services::Status allocateInput() DAAL_C11_OVERRIDE
    {
        return services::Status();
    }

    /**
     * Initializes values of weights and biases if needed
     */
    virtual services::Status initializeInput() DAAL_C11_OVERRIDE
    {
        services::Status s;
        data_management::TensorPtr tensor;

        Parameter *param = getLayerParameter();

        if( !param ) { return services::Status(); }

        tensor = getLayerInput()->get(weights);
        if( tensor && tensor->getDimensions().size() )
        {
            param->weightsInitializer->input.set(initializers::data, tensor);
            if( param->weightsInitializer->parameterBase != NULL )
                param->weightsInitializer->parameterBase->layer  = LayerIfacePtr(this, services::EmptyDeleter());
            s |= param->weightsInitializer->compute();
        }

        tensor = getLayerInput()->get(biases);
        if( tensor && tensor->getDimensions().size() )
        {
            param->biasesInitializer->input.set(initializers::data, tensor);
            if( param->biasesInitializer->parameterBase != NULL )
                param->biasesInitializer->parameterBase->layer = LayerIfacePtr(this, services::EmptyDeleter());
            s |= param->biasesInitializer->compute();
        }

        param->weightsAndBiasesInitialized = true;
        return s;
    }

    /**
     * \copydoc LayerIface::addInput
     */
    virtual services::Status addInput(forward::ResultPtr result, size_t resultIndex, size_t inputIndex) DAAL_C11_OVERRIDE
    {
        return getLayerInput()->addData(result->getValue(resultIndex), inputIndex);
    }
};

typedef services::SharedPtr<LayerIfaceImpl> LayerIfaceImplPtr;
/** @} */
} // namespace interface1
using interface1::LayerIface;
using interface1::LayerIfacePtr;
using interface1::LayerIfaceImpl;
using interface1::LayerIfaceImplPtr;

} // namespace forward
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
