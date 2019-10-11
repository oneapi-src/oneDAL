/* file: eltwise_sum_layer_forward_types.h */
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
//  Implementation of forward element-wise sum layer.
//--
*/

#ifndef __ELTWISE_SUM_LAYER_FORWARD_TYPES_H__
#define __ELTWISE_SUM_LAYER_FORWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "services/daal_defines.h"

#include "data_management/data/tensor.h"

#include "algorithms/neural_networks/layers/layer_forward_types.h"
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
 * @defgroup eltwise_sum_forward Forward Element-wise Sum Layer
 * \copydoc daal::algorithms::neural_networks::layers::eltwise_sum::forward
 * @ingroup eltwise_sum
 * @{
 */
/**
 * \brief Contains classes for the forward element-wise sum layer
 */
namespace forward
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELTWISE_SUM__FORWARD__INPUTID"></a>
 * Available identifiers of input objects for the element-wise sum layer
 */
enum InputId
{
    coefficients = layers::forward::lastInputLayerDataId + 1, /*!< Input coefficients */
    lastInputId  = coefficients
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELTWISE_SUM__FORWARD__INPUT"></a>
 * \brief %Input objects for the forward element-wise sum layer
 * \DAAL_DEPRECATED
 */
class DAAL_EXPORT Input : public layers::forward::Input
{
public:
    typedef layers::forward::Input super;
    /**
     * Default constructo
     * \DAAL_DEPRECATED
     */
    Input();

    /**
     * Copy constructor
     * \DAAL_DEPRECATED
     */
    Input(const Input& other);

    /*
     * \DAAL_DEPRECATED
     */
    virtual ~Input() {}

    /**
     * Sets an input object for the forward element-wise sum layer
     */
    using layers::forward::Input::set;
    /**
     * Returns an input object for the forward element-wise sum layer
    */
    using layers::forward::Input::get;

    /**
    * Returns an input tensor of the forward element-wise sum layer
    * \param[in] id Identifier of the input tensor
    * \return       Input tensor that corresponds to the given identifier
    * \DAAL_DEPRECATED
    */
    DAAL_DEPRECATED data_management::TensorPtr get(InputId id) const;

    /**
    * Sets an input tensor of the forward element-wise sum layer
    * \param[in] id    Identifier of the input tensor
    * \param[in] value Pointer to the tensor
    * \DAAL_DEPRECATED
    */
    DAAL_DEPRECATED void set(InputId id, const data_management::TensorPtr &value);

    /**
    * Returns an input tensor of the forward element-wise sum layer
    * \param[in] id    Identifier of the input tensor
    * \param[in] index Index of the input tensor
    * \return          Input tensor that corresponds to the given identifier
    * \DAAL_DEPRECATED
    */
    DAAL_DEPRECATED data_management::TensorPtr get(layers::forward::InputLayerDataId id, size_t index) const;

    /**
    * Sets an input tensor for the forward element-wise sum layer
    * \param[in] id    Identifier of the input tensor
    * \param[in] value Pointer to the tensor
    * \param[in] index Index of the input tensor
    * \DAAL_DEPRECATED
    */
    DAAL_DEPRECATED void set(layers::forward::InputLayerDataId id, const data_management::TensorPtr &value, size_t index);

    /**
     * Adds tensor with data to the input object of the forward element-wise sum layer
     * \param[in] dataTensor Tensor with data
     * \param[in] index      Index of the tensor with data
     *
     * \return Status of computations
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED_VIRTUAL virtual services::Status addData(const data_management::TensorPtr &dataTensor, size_t index) DAAL_C11_OVERRIDE;

    /**
     * Erases input data tensor from the input of the forward layer
     *
     * \return Status of computations
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED_VIRTUAL virtual services::Status eraseInputData() DAAL_C11_OVERRIDE;

    /**
     * Checks input object of the forward element-wise sum layer
     * \param[in] parameter %Parameter of layer
     * \param[in] method    Computation method of the layer
     *
     * \return Status of computations
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED services::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

private:
    /**
     * \private
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED services::Status checkInputTensors(const LayerData &layerData) const;
    /**
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED services::Status checkCoefficients(const LayerData &layerData) const;

    /**
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED LayerDataPtr getInputLayerDataAllocateIfEmpty();
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELTWISE_SUM__FORWARD__RESULT"></a>
 * \brief Results obtained with the compute() method of the forward element-wise sum layer
 *        in the batch processing mode
 * \DAAL_DEPRECATED
 */
class DAAL_EXPORT Result : public layers::forward::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);

    /**
     * Default constructor
     * \DAAL_DEPRECATED
     */
    Result();

    virtual ~Result() {}

    /**
     * Returns the result of the forward element-wise sum layer
     */
    using layers::forward::Result::get;

    /**
     * Sets the result of the forward element-wise sum layer
     */
    using layers::forward::Result::set;

    /**
     * Allocates memory to store the result of forward  element-wise sum layer
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of forward element-wise sum layer
     * \param[in] method    Computation method for the layer
     *
     * \return Status of computations
     * \DAAL_DEPRECATED
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input,
                                          const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Returns dimensions of value tensor
     * \return Dimensions of value tensor
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED_VIRTUAL virtual const services::Collection<size_t> getValueSize(const services::Collection<size_t> &inputSize,
                                                            const daal::algorithms::Parameter *par, const int method) const DAAL_C11_OVERRIDE;

    /**
    * Returns collection of dimensions of element-wise sum layer output
    * \param[in] inputSize   Collection of input tensors dimensions
    * \param[in] parameter   Parameters of the algorithm
    * \param[in] method      Method of the algorithm
    * \return    Collection of dimensions of element-wise sum layer output
     * \DAAL_DEPRECATED
    */
    DAAL_DEPRECATED_VIRTUAL virtual services::Collection<size_t> getValueSize(const services::Collection< services::Collection<size_t> > &inputSize,
                                                      const daal::algorithms::Parameter *parameter, const int method) DAAL_C11_OVERRIDE;

    /**
     * Returns the result tensor of forward element-wise sum layer
     * \param[in] id Identifier of the result tensor
     * \return       Result tensor that corresponds to the given identifier
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED data_management::TensorPtr get(LayerDataId id) const;

    /**
     * Returns the result numeric table of the forward element-wise sum layer
     * \param[in] id Identifier of the result numeric table
     * \return       Result numeric table that corresponds to the given identifier
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED data_management::NumericTablePtr get(LayerDataNumericTableId id) const;

    /**
     * Sets the result tensor of forward element-wise sum layer
     * \param[in] id    Identifier of the result tensor
     * \param[in] value Result tensor
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED void set(LayerDataId id, const data_management::TensorPtr &value);

    /**
     * Sets the result numeric table of the forward element-wise sum layer
     * \param[in] id    Identifier of the result numeric table
     * \param[in] value Result numeric tensor
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED void set(LayerDataNumericTableId id, const data_management::NumericTablePtr &value);

    /**
     * Sets the result that is used in backward layer
     * \param[in] input  Pointer to an object containing the input data
     *
     * \return Status of operation
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED_VIRTUAL virtual services::Status setResultForBackward(const daal::algorithms::Input *input) DAAL_C11_OVERRIDE;

    /**
     * Checks the result of the forward element-wise sum layer
     * \param[in] input   %Input object of the layer
     * \param[in] par     %Parameter of the layer
     * \param[in] method  Computation method of the layer
     *
     * \return Status of computations
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED_VIRTUAL virtual services::Status check(const daal::algorithms::Input *input,
                                   const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

protected:
    /**
     * \private
     * \DAAL_DEPRECATED
     */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }

private:
    /**
     * \private
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED services::Status checkValue(const Input *input) const;
    /*
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED services::Status checkAuxCoefficients(const Input *input) const;
    /*
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED services::Status checkAuxNumberOfCoefficients(const Input *input) const;

    /*
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED LayerDataPtr getResultLayerDataAllocateIfEmpty();

    /*
     * \DAAL_DEPRECATED
     */
    template<typename algorithmFPType>
    DAAL_DEPRECATED services::Status allocateValueTensor(const Input *eltwiseInput);
};

typedef services::SharedPtr<Result> ResultPtr;
} // namespace interface1

using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;

} // namespace forward
/** @} */
} // namespace eltwise_sum
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
