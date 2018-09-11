/* file: batch_normalization_layer_forward_types.h */
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
//  Implementation of the forward batch normalization layer.
//--
*/

#ifndef __BATCH_NORMALIZATION_LAYER_FORWARD_TYPES_H__
#define __BATCH_NORMALIZATION_LAYER_FORWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"
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
 * @defgroup batch_normalization_forward Forward Batch Normalization Layer
 * \copydoc daal::algorithms::neural_networks::layers::batch_normalization::forward
 * @ingroup batch_normalization
 * @{
 */
/**
 * \brief Contains classes for forward batch normalization layer
 */
namespace forward
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCH_NORMALIZATION__FORWARD__INPUTLAYERDATAID"></a>
 * Available identifiers of input objects for the forward batch normalization layer
 */
enum InputLayerDataId
{
    populationMean = layers::forward::lastInputLayerDataId + 1, /*!< 1-dimensional tensor of size \f$n_k\f$ that stores population mean computed on the previous stage */
    populationVariance,                                    /*!< 1-dimensional tensor of size \f$n_k\f$ that stores population variance computed on the previous stage */
    lastInputLayerDataId = populationVariance
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCH_NORMALIZATION__FORWARD__INPUT"></a>
 * \brief %Input objects for the forward batch normalization layer.
 */
class DAAL_EXPORT Input : public layers::forward::Input
{
public:
    typedef layers::forward::Input super;
    /** Default constructor */
    Input();

    /** Copy constructor */
    Input(const Input& other);

    virtual ~Input() {}

    /**
     * Returns an input object for the forward batch normalization layer
     */
    using layers::forward::Input::get;
    /**
    * Sets an input object for the forward batch normalization layer
    */
    using layers::forward::Input::set;


    /**
     * Allocates memory to store input objects of forward batch normalization layer
     * \param[in] parameter %Parameter of forward batch normalization layer
     * \param[in] method    Computation method for the layer
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Parameter *parameter, const int method);

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

    /**
     * Returns an input object for forward batch normalization layer
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::TensorPtr get(InputLayerDataId id) const;

    /**
     * Sets input for the forward batch normalization layer
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Input object to set
     */
    void set(InputLayerDataId id, const data_management::TensorPtr &ptr);

    /**
     * Checks input object of the forward batch normalization layer
     * \param[in] parameter %Parameter of layer
     * \param[in] method    Computation method of the layer
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCH_NORMALIZATION__FORWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the forward batch normalization layer
 */
class DAAL_EXPORT Result : public layers::forward::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);
    /** Default Constructor */
    Result();
    virtual ~Result() {}

    using layers::forward::Result::get;
    using layers::forward::Result::set;

    /**
     * Returns dimensions of value tensor
     * \return Dimensions of value tensor
     */
    virtual const services::Collection<size_t> getValueSize(const services::Collection<size_t> &inputSize,
                                                            const daal::algorithms::Parameter *par, const int method) const DAAL_C11_OVERRIDE;

    /**
     * Allocates memory to store the result of the forward batch normalization layer
     * \param[in] input Pointer to an object containing the input data
     * \param[in] parameter %Parameter of the forward batch normalization layer
     * \param[in] method Computation method for the layer
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Sets the result that is used in backward batch normalization layer
     * \param[in] input     Pointer to an object containing the input data
     *
     * \return Status of computations
     */
    virtual services::Status setResultForBackward(const daal::algorithms::Input *input) DAAL_C11_OVERRIDE;

    /**
     * Returns the result of the forward batch normalization layer
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::TensorPtr get(LayerDataId id) const;

    /**
     * Sets the result of the forward batch normalization layer
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Result
     */
    void set(LayerDataId id, const data_management::TensorPtr &ptr);

    /**
     * Checks the result of the forward batch normalization layer
     * \param[in] input     %Input of the layer
     * \param[in] parameter %Parameter of the layer
     * \param[in] method    Computation method of the layer
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

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

} // namespace forward

/** @} */
} // namespace batch_normalization
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif
