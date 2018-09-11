/* file: layer_backward.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
//  Implementation of neural network layer.
//--
*/

#ifndef __LAYER_BACKWARD_H__
#define __LAYER_BACKWARD_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_backward_types.h"

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
 * \brief Contains classes for the backward stage of the neural network layer
 */
namespace backward
{
namespace interface1
{
/**
 * @ingroup layers_backward
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BACKWARD__LAYERIFACE"></a>
 *  \brief Abstract class which defines interface for the layer
 */
class LayerIface : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::neural_networks::layers::backward::Input  InputType;
    typedef algorithms::neural_networks::layers::Parameter        ParameterType;
    typedef algorithms::neural_networks::layers::backward::Result ResultType;

    virtual ~LayerIface() {};

    /**
     * Returns the structure that contains results of the layer
     * \return Structure that contains results of the layer
     */
    virtual backward::ResultPtr getLayerResult() = 0;

    /**
     * Returns the structure that contains input objects of the layer
     * \return Structure that contains input objects of the layer
     */
    virtual InputType *getLayerInput() = 0;

    /**
     * Returns the structure that contains parameters of the layer
     * \return Structure that contains parameters of the layer
     */
    virtual ParameterType *getLayerParameter() = 0;

    /**
     * Returns a pointer to the newly allocated backward neural network layer with a copy of input objects
     * and parameters of this layer
     * \return Pointer to the newly allocated backward layer
     */
    services::SharedPtr<daal::algorithms::neural_networks::layers::backward::interface1::LayerIface> clone() const
    {
        return services::SharedPtr<LayerIface>(cloneImpl());
    }

    /**
     * Allocates memory buffers needed for the computations
     */
    virtual services::Status allocateResult() = 0;

    /**
     * Connects two layers in neural network by getting tensor with gradient
     * from the result of the previous layer and adding it to the input object of this layer algorithm
     * \param[in] result        Structure that contains results of the previous layer
     * \param[in] resultIndex   Index of the tensor with gradient in the structure that contains
     *                          results of the previous layer
     * \param[in] inputIndex    Index in the input object of this layer algorithm
     *                          where the tensor with gradient should be placed
     */
     virtual services::Status addInput(backward::ResultPtr result, size_t resultIndex, size_t inputIndex) = 0;

protected:
    virtual LayerIface *cloneImpl() const = 0;
};

typedef services::SharedPtr<LayerIface> LayerIfacePtr;
/** @} */
/**
 * @ingroup layers_backward
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BACKWARD__LAYERIFACE"></a>
 *  \brief Implements the abstract interface LayerIface. LayerIfaceImpl is, in turn, the base class
 *         for the classes interfacing the layers.
 */
class LayerIfaceImpl : public LayerIface
{
public:
    typedef LayerIface super;

    typedef super::InputType     InputType;
    typedef super::ParameterType ParameterType;
    typedef super::ResultType    ResultType;

    virtual ~LayerIfaceImpl() {};

    /**
     * \copydoc LayerIface::addInput
     */
    virtual services::Status addInput(backward::ResultPtr result, size_t resultIndex, size_t inputIndex) DAAL_C11_OVERRIDE
    {
        return getLayerInput()->addInputGradient(result->getGradient(resultIndex), inputIndex);
    }
};

typedef services::SharedPtr<LayerIfaceImpl> LayerIfaceImplPtr;
/** @} */
} // namespace interface1

using interface1::LayerIface;
using interface1::LayerIfacePtr;
using interface1::LayerIfaceImpl;
using interface1::LayerIfaceImplPtr;
} // namespace backward
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
