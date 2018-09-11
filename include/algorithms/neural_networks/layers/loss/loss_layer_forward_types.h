/* file: loss_layer_forward_types.h */
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
//  Implementation of the forward loss layer types.
//--
*/

#ifndef __LOSS_LAYER_FORWARD_TYPES_H__
#define __LOSS_LAYER_FORWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace loss
{
/**
 * @defgroup loss_forward Forward Loss Layer
 * \copydoc daal::algorithms::neural_networks::layers::loss::forward
 * @ingroup loss
 * @{
 */
/**
 * \brief Contains classes for the forward loss layer
 */
namespace forward
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__FORWARD__INPUTID"></a>
 * Available identifiers of input objects for the loss layer algorithm
 */
enum InputId
{
    data        = layers::forward::data,       /*!< Input data */
    weights     = layers::forward::weights,    /*!< Weights of the neural network layer */
    biases      = layers::forward::biases,      /* Biases of the neural network layer */
    groundTruth = layers::forward::lastInputLayerDataId + 1,   /*!< Ground truth for the loss layer */
    lastInputId = groundTruth
};
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__FORWARD__INPUT"></a>
 * \brief %Input objects for the forward loss layer
 */
class DAAL_EXPORT Input : public layers::forward::Input
{
public:
    typedef layers::forward::Input super;
    /** Default constructor */
    Input(size_t nElements = lastInputId + 1);

    /**
     * Returns an input object for the forward loss layer
     */
    using layers::forward::Input::get;

    /**
     * Sets an input object for the forward loss layer
     */
    using layers::forward::Input::set;

    /**
     * Returns input Tensor of the loss layer algorithm
     * \param[in] id    Identifier of the input tensor
     * \return          %Input tensor that corresponds to the given identifier
     */
    data_management::TensorPtr get(forward::InputId id) const;

    /**
     * Sets input for the loss layer algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::TensorPtr &ptr);

    /**
     * Returns dimensions of weights tensor
     * \return Dimensions of weights tensor
     */
    virtual const services::Collection<size_t> getWeightsSizes(const layers::Parameter *parameter) const DAAL_C11_OVERRIDE;

    /**
     * Returns dimensions of biases tensor
     * \return Dimensions of biases tensor
     */
    virtual const services::Collection<size_t> getBiasesSizes(const layers::Parameter *parameter) const DAAL_C11_OVERRIDE;

    virtual ~Input() {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__FORWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the forward loss layer
 */
class DAAL_EXPORT Result : public layers::forward::Result
{
public:
    /** Default constructor */
    Result();
    virtual ~Result() {};

    /**
     * Returns the result of the forward loss layer
     */
    using layers::forward::Result::get;

    /**
     * Sets the result of the forward loss layer
     */
    using layers::forward::Result::set;

    /**
     * Checks the result of the forward loss layer
     * \param[in] input   %Input object for the loss layer
     * \param[in] par     %Parameter of the loss layer
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    /**
     * Allocates memory to store the result of the forward loss layer
     * \param[in] input Pointer to an object containing the input data
     * \param[in] method Computation method for the loss layer
     * \param[in] parameter %Parameter of the forward loss layer
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Returns dimensions of value tensor
     * \return Dimensions of value tensor
     */
    virtual const services::Collection<size_t> getValueSize(const services::Collection<size_t> &inputSize,
            const daal::algorithms::Parameter *par, const int method) const DAAL_C11_OVERRIDE;
};
} // namespace interface1
using interface1::Input;
using interface1::Result;
} // namespace forward
/** @} */
} // namespace loss
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
