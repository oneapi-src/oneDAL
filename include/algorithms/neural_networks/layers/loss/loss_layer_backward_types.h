/* file: loss_layer_backward_types.h */
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
//  Implementation of the backward loss layer types.
//--
*/

#ifndef __LOSS_LAYER_BACKWARD_TYPES_H__
#define __LOSS_LAYER_BACKWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_backward_types.h"

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
 * @defgroup loss_backward Backward Loss Layer
 * \copydoc daal::algorithms::neural_networks::layers::loss::backward
 * @ingroup loss
 * @{
 */
/**
 * \brief Contains classes for the backward loss layer
 */
namespace backward
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__BACKWARD__INPUT"></a>
 * \brief %Input objects for the backward loss layer
 */
class Input : public layers::backward::Input
{
public:
    /** Default constructor */
    Input() {};

    virtual ~Input() {}

    /**
     * Returns an input object for the backward loss layer
     */
    using layers::backward::Input::get;

    /**
     * Sets an input object for the backward loss layer
     */
    using layers::backward::Input::set;

    /**
     * Checks an input object for the backward loss layer
     * \param[in] par     Algorithm parameter
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }
        services::SharedPtr<LayerData> layerData = get(layers::backward::inputFromForward);
        if (!layerData) { this->_errors->add(services::ErrorNullLayerData); return; }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__BACKWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the backward loss layer
 */
class Result : public layers::backward::Result
{
public:
    /** Default constructor */
    Result() : layers::backward::Result() {};

    virtual ~Result() {};

    /**
     * Returns an result object for the backward loss layer
     */
    using layers::backward::Result::get;

    /**
     * Sets an result object for the backward loss layer
     */
    using layers::backward::Result::set;

    /**
     * Checks the result of the backward loss layer
     * \param[in] input   %Input object for the layer
     * \param[in] par     %Parameter of the layer
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {}
};
} // namespace interface1
using interface1::Input;
using interface1::Result;
} // namespace backward
/** @} */
} // namespace loss
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
