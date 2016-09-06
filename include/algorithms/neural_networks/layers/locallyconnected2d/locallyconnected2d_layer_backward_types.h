/* file: locallyconnected2d_layer_backward_types.h */
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
//  Implementation of backward two-dimensional (2D) locally connected layer.
//--
*/

#ifndef __LOCALLYCONNECTED2D_LAYER_BACKWARD_TYPES_H__
#define __LOCALLYCONNECTED2D_LAYER_BACKWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_backward_types.h"
#include "algorithms/neural_networks/layers/locallyconnected2d/locallyconnected2d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace locallyconnected2d
{
/**
 * @defgroup locallyconnected2d_backward Backward Two-dimensional __onvolution Layer
 * \copydoc daal::algorithms::neural_networks::layers::locallyconnected2d::backward
 * @ingroup locallyconnected2d
 * @{
 */
/**
 * \brief Contains classes for the backward 2D locally connected layer
 */
namespace backward
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOCALLYCONNECTED2D__BACKWARD__INPUT"></a>
 * \brief %Input objects for the backward 2D locally connected layer
 */
class DAAL_EXPORT Input : public layers::backward::Input
{
public:
    /**
     * Default constructor
     */
    Input();

    virtual ~Input() {}

    /**
     * Sets an input object for the backward 2D locally connected layer
     */
    using layers::backward::Input::get;

    /**
     * Returns an input object for the backward 2D locally connected layer
    */
    using layers::backward::Input::set;

    /**
     * Returns an input object for backward 2D locally connected layer
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::TensorPtr get(LayerDataId id) const;

    /**
     * Sets input for the backward 2D locally connected layer
     * \param[in] id    Identifier of the input  object
     * \param[in] value Input object to set
     */
    void set(LayerDataId id, const data_management::TensorPtr &value);

    /**
     * Checks an input object of the 2D locally connected layer
     * \param[in] parameter %Parameter of layer
     * \param[in] method    Computation method of the layer
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOCALLYCONNECTED2D__BACKWARD__RESULT"></a>
 * \brief Results obtained with the compute() method of the backward 2D locally connected layer
 */
class DAAL_EXPORT Result : public layers::backward::Result
{
public:
    /**
     * Default constructor
     */
    Result();

    virtual ~Result() {}

    /**
     * Returns the result of the backward 2D locally connected layer
     */
    using layers::backward::Result::get;

    /**
     * Sets the result of the backward 2D locally connected layer
     */
    using layers::backward::Result::set;

    /**
     * Allocates memory to store the result of backward 2D locally connected layer
     * \param[in] input     Object containing the input data
     * \param[in] parameter %Parameter of backward 2D locally connected layer
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    void DAAL_EXPORT allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Checks the result of the 2D locally connected layer
     * \param[in] input   %Input object of the layer
     * \param[in] par     %Parameter of the layer
     * \param[in] method  Computation method of the layer
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

protected:
    void getBiasesDims(const Input *algInput, const Parameter *param, services::Collection<size_t> &bDims) const;
};

} // namespace interface1
using interface1::Input;
using interface1::Result;
} // namespace backward
/** @} */
} // namespace locallyconnected2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
