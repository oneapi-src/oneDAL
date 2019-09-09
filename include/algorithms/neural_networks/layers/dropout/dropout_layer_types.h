/* file: dropout_layer_types.h */
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
//  Implementation of the dropout layer types.
//--
*/

#ifndef __DROPOUT_LAYER_TYPES_H__
#define __DROPOUT_LAYER_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_types.h"
#include "algorithms/engines/mt19937/mt19937.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * @defgroup dropout Dropout Layer
 * \copydoc daal::algorithms::neural_networks::layers::dropout
 * @ingroup layers
 * @{
 */
namespace dropout
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__DROPOUT__METHOD"></a>
 * \brief Computation methods for the dropout layer
 */
enum Method
{
    defaultDense = 0, /*!<  Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__DROPOUT__LAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward dropout layer and results for the forward dropout layer
 */
enum LayerDataId
{
    auxRetainMask = layers::lastLayerInputLayout + 1, /*!< Tensor filled with Bernoulli random variates  (0 in positions that are dropped,
                                                                 1 - in the others) divided by probability that any particular element is retained. */
    lastLayerDataId = auxRetainMask
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__DROPOUT__PARAMETER"></a>
 * \brief Parameters for the dropout layer
 *
 * \snippet neural_networks/layers/dropout/dropout_layer_types.h Parameter source code
 */
/* [Parameter source code] */
class DAAL_EXPORT Parameter: public layers::Parameter
{
public:
    /**
    *  Constructs parameters of the dropout layer
    *  \param[in] retainRatio_ Probability that any particular element is retained
    *  \param[in] seed_        Seed for mask elements random generation \DAAL_DEPRECATED_USE{ engine }
    */
    Parameter(const double retainRatio_ = 0.5, const size_t seed_ = 777);

    double retainRatio;         /*!< Probability that any particular element is retained. */
    size_t seed;                /*!< Seed for mask elements random generation. \DAAL_DEPRECATED_USE{ engine } */
    engines::EnginePtr engine;  /*!< Engine for mask elements random generation. */
    /**
     * Checks the correctness of the parameter
     *
     * \return Status of computations
     */
    virtual services::Status check() const;
};
/* [Parameter source code] */

} // namespace interface1
using interface1::Parameter;

} // namespace dropout
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
