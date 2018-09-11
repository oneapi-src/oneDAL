/* file: split_layer_backward_types.h */
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
//  Implementation of the backward split layer
//--
*/

#ifndef __SPLIT_LAYER_BACKWARD_TYPES_H__
#define __SPLIT_LAYER_BACKWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_backward_types.h"
#include "algorithms/neural_networks/layers/split/split_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * \brief Contains classes for the split layer
 */
namespace split
{
/**
 * @defgroup split_backward Backward Split Layer
 * \copydoc daal::algorithms::neural_networks::layers::split::backward
 * @ingroup split
 * @{
 */
/**
 * \brief Contains classes for the backward split layer
 */
namespace backward
{
/**
* <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPLIT__BACKWARD__INPUTLAYERDATAID"></a>
* Available identifiers of input objects for the backward split layer
*/
enum InputLayerDataId
{
    inputGradientCollection = 1   /*!< Input structure retrieved from the result of the forward split layer */
};

/**
* \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
*/
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPLIT__BACKWARD__INPUT"></a>
 * \brief %Input parameters for the backward split layer
 */
class DAAL_EXPORT Input : public layers::backward::Input
{
public:
    /** \brief Default constructor */
    Input();

    virtual ~Input() {}

    /**
     * Returns an input object for the backward split layer
     */
    using layers::backward::Input::get;

    /**
     * Sets an input object for the backward split layer
     */
    using layers::backward::Input::set;

    /**
     * Returns a tensor with a given index from the collection of input tensors
     * \param[in] id    Identifier of the collection of input tensors
     * \param[in] index Index of the tensor to be returned
     * \return          Pointer to the table with the input tensor
     */
    data_management::TensorPtr get(InputLayerDataId id, size_t index) const;

    /**
     * Returns input Tensor of the layer algorithm
     * \param[in] id    Identifier of the input tensor
     * \return          %Input tensor that corresponds to the given identifier
     */
    LayerDataPtr get(InputLayerDataId id) const;

    /**
     * Sets an input object for the backward split layer
     * \param[in] id     Identifier of the input object
     * \param[in] value  Pointer to the input object
     * \param[in] index  Index of the tensor to be set
     */
    void set(InputLayerDataId id, const data_management::TensorPtr &value, size_t index);

    /**
    * Sets input for the layer algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(InputLayerDataId id, const LayerDataPtr &ptr);

    /**
     * Adds tensor with input gradient to the input object of the backward split layer
     * \param[in] igTensor  Tensor with input gradient
     * \param[in] index     Index of the tensor with input gradient
     *
     * \return Status of computations
     */
    virtual services::Status addInputGradient(const data_management::TensorPtr &igTensor, size_t index) DAAL_C11_OVERRIDE;

    /**
     * Sets input structure retrieved from the result of the forward layer
     * \param[in] result Result of the forward layer
     */
    virtual services::Status setInputFromForward(layers::forward::ResultPtr result) DAAL_C11_OVERRIDE;

    /**
     * Checks an input object of the backward split layer
     * \param[in] par     Algorithm parameter
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    /**
    * Returns the layout of the input object for the layer algorithm
    * \return Layout of the input object for the layer algorithm
    */
    virtual LayerInputLayout getLayout() const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPLIT__BACKWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method of the backward split layer
 */
class DAAL_EXPORT Result : public layers::backward::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);
    /** \brief Default constructor */
    Result();
    virtual ~Result() {};

    /**
     * Returns the result of the backward split layer
     */
    using layers::backward::Result::get;

    /**
     * Sets the result of the backward split layer
     */
    using layers::backward::Result::set;

    /**
     * Checks the result of the backward split layer
     * \param[in] input   %Input object for the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    /**
    * Allocates memory to store the result of the backward split layer
     * \param[in] input     Pointer to an object containing the input data
     * \param[in] method    Computation method for the algorithm
     * \param[in] parameter %Parameter of the backward split layer
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;

} // namespace interface1
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;
} // namespace backward
/** @} */
} // namespace split
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
