/* file: pooling2d_layer_backward_types.h */
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
//  Implementation of backward 2D pooling layer.
//--
*/

#ifndef __POOLING2D_LAYER_BACKWARD_TYPES_H__
#define __POOLING2D_LAYER_BACKWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_backward_types.h"
#include "algorithms/neural_networks/layers/pooling2d/pooling2d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace pooling2d
{
/**
 * @defgroup pooling2d_backward Backward Two-dimensional Pooling Layer
 * \copydoc daal::algorithms::neural_networks::layers::pooling2d::backward
 * @ingroup pooling2d
 * @{
 */
/**
 * \brief Contains classes for backward one-dimensional (2D) pooling layer
 */
namespace backward
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING2D__BACKWARD__INPUT"></a>
 * \brief %Input objects for the backward 2D pooling layer
 */
class Input : public layers::backward::Input
{
public:
    /** Default constructor */
    Input() {}

    virtual ~Input() {}

    /**
     * Checks an input object for the backward 2D pooling layer
     * \param[in] parameter Algorithm parameter
     * \param[in] method Computation method
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        const Parameter *param = static_cast<const Parameter *>(parameter);

        data_management::TensorPtr inputGradientTensor = get(layers::backward::inputGradient);
        if (!data_management::checkTensor(inputGradientTensor.get(), this->_errors.get(), inputGradientStr())) { return; }

        size_t nDim = inputGradientTensor->getNumberOfDimensions();
        if(param->indices.size[0] > nDim - 1 || param->indices.size[1] > nDim - 1 ||
           param->indices.size[0] == param->indices.size[1])
        {
            services::SharedPtr<services::Error> error = services::SharedPtr<services::Error>(new services::Error());
            error->setId(services::ErrorIncorrectParameter);
            error->addStringDetail(services::ArgumentName, "indices");
            this->_errors->add(error);
            return;
        }

        services::Collection<size_t> inputDims = getInputGradientSize(param);

        if (!data_management::checkTensor(inputGradientTensor.get(), this->_errors.get(), inputGradientStr(), &inputDims)) { return; }
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

    virtual services::Collection<size_t> getInputGradientSize(const pooling2d::Parameter *parameter) const
    {
        const Parameter *param = static_cast<const Parameter *>(parameter);
        services::Collection<size_t> inputDims = getGradientSize();

        for (size_t d = 0; d < 2; d++)
        {
            inputDims[param->indices.size[d]] = computeInputDimension(
                inputDims[param->indices.size[d]], param->kernelSizes.size[d], param->paddings.size[d], param->strides.size[d]);
        }
        return inputDims;
    }

    size_t computeInputDimension(size_t maskDim, size_t kernelSize, size_t padding, size_t stride) const
    {
        size_t inputDim = (maskDim + 2 * padding - kernelSize + stride) / stride;
        return inputDim;
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING2D__BACKWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the backward 2D pooling layer
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
            DAAL_ALLOCATE_TENSOR_AND_SET(layers::backward::gradient, in->getGradientSize());
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

        for (size_t i = 0; i < 2; i++)
        {
            size_t index = param->indices.size[i];
            size_t kernelSize = param->kernelSizes.size[i];

            if (kernelSize == 0 || kernelSize > gradientDims[index] + 2 * param->paddings.size[i])
            {
                services::SharedPtr<services::Error> error = services::SharedPtr<services::Error>(new services::Error());
                error->setId(services::ErrorIncorrectParameter);
                error->addStringDetail(services::ArgumentName, "kernelSize");
                this->_errors->add(error);
                return;
            }
        }
    }
};

} // namespace interface1
using interface1::Input;
using interface1::Result;
} // namespace backward
/** @} */
} // namespace pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif
