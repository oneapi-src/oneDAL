/* file: spatial_pooling2d_layer_backward_types.h */
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
//  Implementation of backward 2D spatial layer.
//--
*/

#ifndef __SPATIAL_POOLING2D_LAYER_BACKWARD_TYPES_H__
#define __SPATIAL_POOLING2D_LAYER_BACKWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_backward_types.h"
#include "algorithms/neural_networks/layers/spatial_pooling2d/spatial_pooling2d_layer_types.h"
#include "algorithms/neural_networks/layers/spatial_pooling2d/spatial_pooling2d_layer_forward_types.h"

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
/**
 * @defgroup spatial_pooling2d_backward Backward Two-dimensional Spatial Pyramid Pooling Layer
 * \copydoc daal::algorithms::neural_networks::layers::spatial_pooling2d::backward
 * @ingroup spatial_pooling2d
 * @{
 */
/**
 * \brief Contains classes for backward one-dimensional (2D) spatial layer
 */
namespace backward
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_POOLING2D__BACKWARD__INPUT"></a>
 * \brief %Input objects for the backward 2D spatial layer
 */
class Input : public layers::backward::Input
{
public:
    /** Default constructor */
    Input() {}

    virtual ~Input() {}

    using layers::backward::Input::get;
    using layers::backward::Input::set;


    /**
    * Checks an input object for the backward 2D pooling layer
    * \param[in] parameter Algorithm parameter
    * \param[in] method Computation method
    */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        const Parameter *param = static_cast<const Parameter *>(parameter);

        DAAL_CHECK_EX(get(layers::backward::inputGradient)->getDimensions().size() == 2, services::ErrorIncorrectNumberOfDimensionsInTensor, services::ArgumentName, inputGradientStr());

        data_management::TensorPtr inputGradientTensor = get(layers::backward::inputGradient);
        if (!data_management::checkTensor(inputGradientTensor.get(), this->_errors.get(), inputGradientStr())) { return; }

        size_t nDim = inputGradientTensor->getNumberOfDimensions();
        DAAL_CHECK(nDim == 2, ErrorIncorrectParameter);
    }

    /**
     * Return the collection with gradient size
     * \return The collection with gradient size
     */
    virtual services::Collection<size_t> getGradientSize() const
    {
        services::Collection<size_t> dims;
        const data_management::NumericTablePtr inputDims = getAuxInputDimensions();
        if (!inputDims)
        { this->_errors->add(services::ErrorNullInputNumericTable); return dims; }

        data_management::BlockDescriptor<int> block;
        inputDims->getBlockOfRows(0, 1, data_management::readOnly, block);
        int *inputDimsArray = block.getBlockPtr();
        for(size_t i = 0; i < inputDims->getNumberOfColumns(); i++)
        {
            dims.push_back((size_t) inputDimsArray[i]);
        }
        inputDims->releaseBlockOfRows(block);
        return dims;
    }

protected:
    virtual data_management::NumericTablePtr getAuxInputDimensions() const = 0;

    size_t computeInputDimension(size_t maskDim, size_t kernelSize, size_t padding, size_t stride) const
    {
        size_t inputDim = (maskDim + 2 * padding - kernelSize + stride) / stride;
        return inputDim;
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_POOLING2D__BACKWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the backward 2D spatial layer
 */
class Result : public layers::backward::Result
{
public:
    /** Default constructor */
    Result() {}
    virtual ~Result() {}

    /**
     * Allocates memory to store the result of the backward 2D pooling layer
     * \param[in] input Pointer to an object containing the input data
     * \param[in] method Computation method for the layer
     * \param[in] parameter %Parameter of the backward 2D pooling layer
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Input *in = static_cast<const Input *>(input);

        if (!data_management::checkTensor(in->get(layers::backward::inputGradient).get(), this->_errors.get(), inputGradientStr())) { return; }

        if (!get(layers::backward::gradient))
        {
            set(layers::backward::gradient, data_management::TensorPtr(
                new data_management::HomogenTensor<algorithmFPType>(in->getGradientSize(), data_management::Tensor::doAllocate)));
        }
    }

    /**
    * Checks the result of the backward 2D pooling layer
    * \param[in] input %Input object for the layer
    * \param[in] parameter %Parameter of the layer
    * \param[in] method Computation method
    */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        const Parameter *param = static_cast<const Parameter *>(parameter);
        const Input *algInput = static_cast<const Input *>(input);
        const services::Collection<size_t> &gradientDims = algInput->getGradientSize();

        if (!data_management::checkTensor(get(layers::backward::gradient).get(), this->_errors.get(), gradientStr(), &gradientDims)) { return; }

        services::Collection<size_t> valueDims = spatial_pooling2d::forward::Result::computeValueDimensions(get(layers::backward::gradient)->getDimensions(), param);
        DAAL_CHECK(valueDims[1] == algInput->get(layers::backward::inputGradient)->getDimensionSize(1), ErrorIncorrectParameter);
    }

};

} // namespace interface1
using interface1::Input;
using interface1::Result;
} // namespace backward
/** @} */
} // namespace spatial_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif
