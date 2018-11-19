/* file: tanh_layer_backward_types.h */
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
//  Implementation of the backward hyperbolic tangent layer.
//--
*/

#ifndef __TANH_LAYER_BACKWARD_TYPES_H__
#define __TANH_LAYER_BACKWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "algorithms/neural_networks/layers/layer_backward_types.h"
#include "algorithms/neural_networks/layers/tanh/tanh_layer_types.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"


namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * \brief Contains classes for the hyperbolic tangent layer
 */
namespace tanh
{
/**
 * @defgroup tanh_layers_backward Backward Hyperbolic Tangent Layer
 * \copydoc daal::algorithms::neural_networks::layers::tanh::backward
 * @ingroup tanh_layers
 * @{
 */
/**
 * \brief Contains classes for the backward hyperbolic tangent layer
 */
namespace backward
{
/**
* \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
*/
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__TANH__BACKWARD__INPUT"></a>
 * \brief %Input objects for the backward hyperbolic tangent layer
 */
class DAAL_EXPORT Input : public layers::backward::Input
{
public:
    typedef layers::backward::Input super;
    /** \brief Default constructor */
    Input();

    /** Copy constructor */
    Input(const Input& other);

    virtual ~Input() {}

    /**
     * Returns an input object for the backward hyperbolic tangent layer
     */
    using layers::backward::Input::get;
    /**
    * Sets an input object for the backward hyperbolic tangent layer
    */
    using layers::backward::Input::set;

    /**
     * Returns an input object for the backward hyperbolic tangent layer
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::TensorPtr get(LayerDataId id) const;

    /**
     * Sets an input object for the backward hyperbolic tangent layer
     * \param[in] id     Identifier of the input object
     * \param[in] value  Pointer to the input object
     */
    void set(LayerDataId id, const data_management::TensorPtr &value);

    /**
     * Checks an input object of the backward hyperbolic tangent layer
     * \param[in] par     Algorithm parameter
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__TANH__BACKWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method of the backward hyperbolic tangent layer
 */
class DAAL_EXPORT Result : public layers::backward::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);
    /** \brief Default constructor */
    Result();

    virtual ~Result() {};

    /**
     * Returns the result of the backward hyperbolic tangent layer
     */
    using layers::backward::Result::get;
    /**
     * Sets the result of the backward hyperbolic tangent layer
     *
     */
    using layers::backward::Result::set;

    /**
     * Checks the result of the backward hyperbolic tangent layer
     * \param[in] input   %Input object for the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    /**
    * Allocates memory to store the result of the backward hyperbolic tangent layer
     * \param[in] input Pointer to an object containing the input data
     * \param[in] method Computation method for the algorithm
     * \param[in] par %Parameter of the backward layer
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

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
} // namespace tanh
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
