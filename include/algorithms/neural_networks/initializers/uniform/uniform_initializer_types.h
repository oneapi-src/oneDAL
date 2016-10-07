/* file: uniform_initializer_types.h */
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
//  Implementation of uniform initializer.
//--
*/

#ifndef __UNIFORM_INITIALIZER_TYPES_H__
#define __UNIFORM_INITIALIZER_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/initializers/initializer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
/**
 * \brief Contains classes for neural network weights and biases uniform initializer
 */
namespace uniform
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__UNIFORM__METHOD"></a>
 * Available methods to compute uniform initializer
 */
enum Method
{
    defaultDense = 0,    /*!< Default: performance-oriented method. */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__UNIFORM__PARAMETER"></a>
 * \brief Uniform initializer parameters
 */
class Parameter: public initializers::Parameter
{
public:
    /**
     *  Main constructor
     *  \param[in] _a    Left bound a
     *  \param[in] _b    Right bound b
     *  \param[in] _seed Seed for generating random numbers for the initialization
     */
    Parameter(double _a=-0.5, double _b=0.5, size_t _seed=777): a(_a), b(_b), seed(_seed) {}

    double a;    /*!< Left bound a */
    double b;    /*!< Right bound b */
    size_t seed; /*!< Seed for generating random numbers */
};

} // namespace interface1
using interface1::Parameter;

} // namespace uniform
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
