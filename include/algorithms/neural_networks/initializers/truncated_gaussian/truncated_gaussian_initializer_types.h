/* file: truncated_gaussian_initializer_types.h */
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
//  Implementation of truncated gaussian initializer.
//--
*/

#ifndef __TRUNCATED_GAUSSIAN_INITIALIZER_TYPES_H__
#define __TRUNCATED_GAUSSIAN_INITIALIZER_TYPES_H__

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
 * @defgroup initializers_truncated_gaussian Truncated Gaussian Initializer
 * \copydoc daal::algorithms::neural_networks::initializers::truncated_gaussian
 * @ingroup initializers
 * @{
 */
/**
 * \brief Contains classes for neural network weights and biases truncated gaussian initializer
 */
namespace truncated_gaussian
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__TRUNCATED_GAUSSIAN__METHOD"></a>
 * Available methods to compute truncated gaussian initializer
 */
enum Method
{
    defaultDense = 0    /*!< Default: performance-oriented method. */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__TRUNCATED_GAUSSIAN__PARAMETER"></a>
 * \brief truncated gaussian initializer parameters
 */
template<typename algorithmFPType>
class DAAL_EXPORT Parameter : public initializers::Parameter
{
public:
    /**
     *  Main constructor
     *  \param[in] _mean   Mean
     *  \param[in] _sigma  Standard deviation
     *  \param[in] _seed   Seed for generating random numbers for the initialization \DAAL_DEPRECATED_USE{ engine }
     */
    Parameter(double _mean = 0, double _sigma = 1.0, size_t _seed = 777);

    double mean;        /*!< The distribution mean */
    double sigma;       /*!< The standard deviation of the distribution */
    algorithmFPType a;  /*!< Left bound of truncation range */
    algorithmFPType b;  /*!< Right bound of truncation range */
    size_t seed;        /*!< Seed for generating random numbers */

    services::Status check() const DAAL_C11_OVERRIDE;
};

} // namespace interface1
using interface1::Parameter;

} // namespace truncated_gaussian
/** @} */
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
