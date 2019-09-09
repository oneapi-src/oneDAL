/* file: eltwise_sum_layer_backward_types.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of backward element-wise sum layer.
//--
*/

#ifndef __ELTWISE_SUM_LAYER_BACKWARD_TYPES_H__
#define __ELTWISE_SUM_LAYER_BACKWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "services/daal_defines.h"

#include "data_management/data/tensor.h"

#include "algorithms/neural_networks/layers/layer_backward_types.h"
#include "algorithms/neural_networks/layers/eltwise_sum/eltwise_sum_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace eltwise_sum
{
/**
 * @defgroup eltwise_sum_backward Backward Element-wise Sum layer
 * \copydoc daal::algorithms::neural_networks::layers::eltwise_sum::backward
 * @ingroup eltwise_sum
 * @{
 */
/**
 * \brief Contains classes for backward element-wise sum layer
 */
namespace backward
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELTWISE_SUM__BACKWARD__INPUT"></a>
 * \brief %Input objects for the backward element-wise sum layer
 */
class DAAL_EXPORT Input : public layers::backward::Input
{
public:
    typedef layers::backward::Input super;
    /**
     * Default constructor
     */
    Input();

    /** Copy constructor */
    Input(const Input& other);

    virtual ~Input() {}

    /**
     * Sets an input object for the backward element-wise sum layer
     */
    using layers::backward::Input::get;

    /**
     * Returns an input object for the backward element-wise sum layer
    */
    using layers::backward::Input::set;

    /**
     * Returns an input tensor for backward element-twise sum layer
     * \param[in] id Identifier of the input tensor
     * \return       Input tensor that corresponds to the given identifier
     */
    data_management::TensorPtr get(LayerDataId id) const;

    /**
     * Returns an input numeric table for backward element-wise sum layer
     * \param[in] id Identifier of the input numeric table
     * \return       Input numeric table that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(LayerDataNumericTableId id) const;

    /**
     * Sets an input tensor for the backward element-twise sum layer
     * \param[in] id    Identifier of the input tensor
     * \param[in] value Input tensor to set
     */
    void set(LayerDataId id, const data_management::TensorPtr &value);

    /**
     * Sets an input numeric table for the backward element-wise sum layer
     * \param[in] id    Identifier of the input numeric table
     * \param[in] value Input numeric table
     */
    void set(LayerDataNumericTableId id, const data_management::NumericTablePtr &value);

    /**
     * Checks an input object of the element-wise sum layer
     * \param[in] par       %Parameter of layer
     * \param[in] method    Computation method of the layer
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    /**
     * Gets number of coefficients (or number of input tensors on the forward pass)
     *
     * \return Number of coefficients
     */
    size_t getNumberOfCoefficients() const;

private:
    size_t getNumberOfAuxCoefficientsFromTable() const;

    services::Status checkInputGradient() const;
    services::Status checkAuxCoefficients() const;
    services::Status checkAuxNumberOfCoefficients() const;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELTWISE_SUM__BACKWARD__RESULT"></a>
 * \brief Results obtained with the compute() method of the backward element-wise sum layer
 */
class DAAL_EXPORT Result : public layers::backward::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);

    /**
     * Default constructor
     */
    Result();

    virtual ~Result() {}

    /**
     * Returns the result of the backward element-wise sum layer
     */
    using layers::backward::Result::get;

    /**
     * Sets the result of the backward element-wise sum layer
     */
    using layers::backward::Result::set;

    /**
    * Returns the result tensor of the backward element-wise layer
    * \param[in] id    Identifier of the result tensor
    * \param[in] index Index of the result tensor
    * \return          Input tensor that corresponds to the given identifier
    */
    data_management::TensorPtr get(layers::backward::ResultLayerDataId id, size_t index) const;

    /**
     * Sets the result tensor for the backward element-wise layer
     * \param[in] id       Identifier of the result tensor
     * \param[in] value    Pointer to the tensor
     * \param[in] index    Index of the result tensor
     */
    void set(layers::backward::ResultLayerDataId id, const data_management::TensorPtr &value, size_t index);

    /**
     * Returns resulting gradient of the backward element-wise layer
     * \param[in] index Index of the tensor with gradient
     * \return Resulting gradient that corresponds to the given index
     */
    virtual data_management::TensorPtr getGradient(size_t index) const DAAL_C11_OVERRIDE;

    /**
     * Allocates memory to store the result of backward element-wise sum layer
     * \param[in] input     Object containing the input data
     * \param[in] parameter %Parameter of backward element-wise sum layer
     * \param[in] method    Computation method
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input,
                                          const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Checks the result of the element-wise sum layer
     * \param[in] input   %Input object of the layer
     * \param[in] par     %Parameter of the layer
     * \param[in] method  Computation method of the layer
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input *input,
                           const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    /**
     * Returns the layout of the result object for the layer algorithm
     * \return Layout of the result object for the layer algorithm
     */
    virtual LayerResultLayout getLayout() const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }

private:
    template<typename algorithmFPType>
    services::Status allocateNewOutputTensors(const data_management::TensorPtr &inputGradient, size_t nOutputs);

    LayerDataPtr getResultLayerDataAllocateIfEmpty();

    void useInputGradientTensorAsOutput(const data_management::TensorPtr &inputGradient, size_t nOutputs);

    services::Status checkResultLayerData(const Input *input) const;
    services::Status checkOutputGradients(const Input *input) const;
};

typedef services::SharedPtr<Result> ResultPtr;
} // namespace interface1

using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;

} // namespace backward
/** @} */
} // namespace eltwise_sum
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
