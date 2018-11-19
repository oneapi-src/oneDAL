/* file: eltwise_sum_layer_forward_types.h */
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
 */
class DAAL_EXPORT Input : public layers::forward::Input
{
public:
    typedef layers::forward::Input super;
    /**
     * Default constructor
     */
    Input();

    /** Copy constructor */
    Input(const Input& other);

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
    */
    data_management::TensorPtr get(InputId id) const;

    /**
    * Sets an input tensor of the forward element-wise sum layer
    * \param[in] id    Identifier of the input tensor
    * \param[in] value Pointer to the tensor
    */
    void set(InputId id, const data_management::TensorPtr &value);

    /**
    * Returns an input tensor of the forward element-wise sum layer
    * \param[in] id    Identifier of the input tensor
    * \param[in] index Index of the input tensor
    * \return          Input tensor that corresponds to the given identifier
    */
    data_management::TensorPtr get(layers::forward::InputLayerDataId id, size_t index) const;

    /**
    * Sets an input tensor for the forward element-wise sum layer
    * \param[in] id    Identifier of the input tensor
    * \param[in] value Pointer to the tensor
    * \param[in] index Index of the input tensor
    */
    void set(layers::forward::InputLayerDataId id, const data_management::TensorPtr &value, size_t index);

    /**
     * Adds tensor with data to the input object of the forward element-wise sum layer
     * \param[in] dataTensor Tensor with data
     * \param[in] index      Index of the tensor with data
     *
     * \return Status of computations
     */
    virtual services::Status addData(const data_management::TensorPtr &dataTensor, size_t index) DAAL_C11_OVERRIDE;

    /**
     * Erases input data tensor from the input of the forward layer
     *
     * \return Status of computations
     */
    virtual services::Status eraseInputData() DAAL_C11_OVERRIDE;

    /**
     * Checks input object of the forward element-wise sum layer
     * \param[in] parameter %Parameter of layer
     * \param[in] method    Computation method of the layer
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

private:
    /** \private */

    services::Status checkInputTensors(const LayerData &layerData) const;
    services::Status checkCoefficients(const LayerData &layerData) const;

    LayerDataPtr getInputLayerDataAllocateIfEmpty();
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELTWISE_SUM__FORWARD__RESULT"></a>
 * \brief Results obtained with the compute() method of the forward element-wise sum layer
 *        in the batch processing mode
 */
class DAAL_EXPORT Result : public layers::forward::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);

    /**
     * Default constructor
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
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input,
                                          const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Returns dimensions of value tensor
     * \return Dimensions of value tensor
     */
    virtual const services::Collection<size_t> getValueSize(const services::Collection<size_t> &inputSize,
                                                            const daal::algorithms::Parameter *par, const int method) const DAAL_C11_OVERRIDE;

    /**
    * Returns collection of dimensions of element-wise sum layer output
    * \param[in] inputSize   Collection of input tensors dimensions
    * \param[in] parameter   Parameters of the algorithm
    * \param[in] method      Method of the algorithm
    * \return    Collection of dimensions of element-wise sum layer output
    */
    virtual services::Collection<size_t> getValueSize(const services::Collection< services::Collection<size_t> > &inputSize,
                                                      const daal::algorithms::Parameter *parameter, const int method) DAAL_C11_OVERRIDE;

    /**
     * Returns the result tensor of forward element-wise sum layer
     * \param[in] id Identifier of the result tensor
     * \return       Result tensor that corresponds to the given identifier
     */
    data_management::TensorPtr get(LayerDataId id) const;

    /**
     * Returns the result numeric table of the forward element-wise sum layer
     * \param[in] id Identifier of the result numeric table
     * \return       Result numeric table that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(LayerDataNumericTableId id) const;

    /**
     * Sets the result tensor of forward element-wise sum layer
     * \param[in] id    Identifier of the result tensor
     * \param[in] value Result tensor
     */
    void set(LayerDataId id, const data_management::TensorPtr &value);

    /**
     * Sets the result numeric table of the forward element-wise sum layer
     * \param[in] id    Identifier of the result numeric table
     * \param[in] value Result numeric tensor
     */
    void set(LayerDataNumericTableId id, const data_management::NumericTablePtr &value);

    /**
     * Sets the result that is used in backward layer
     * \param[in] input  Pointer to an object containing the input data
     *
     * \return Status of operation
     */
    virtual services::Status setResultForBackward(const daal::algorithms::Input *input) DAAL_C11_OVERRIDE;

    /**
     * Checks the result of the forward element-wise sum layer
     * \param[in] input   %Input object of the layer
     * \param[in] par     %Parameter of the layer
     * \param[in] method  Computation method of the layer
     *
     * \return Status of computations
     */
    virtual services::Status check(const daal::algorithms::Input *input,
                                   const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }

private:
    /** \private */

    services::Status checkValue(const Input *input) const;
    services::Status checkAuxCoefficients(const Input *input) const;
    services::Status checkAuxNumberOfCoefficients(const Input *input) const;

    LayerDataPtr getResultLayerDataAllocateIfEmpty();

    template<typename algorithmFPType>
    services::Status allocateValueTensor(const Input *eltwiseInput);
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
