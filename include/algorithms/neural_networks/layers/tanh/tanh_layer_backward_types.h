/* file: tanh_layer_backward_types.h */
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
