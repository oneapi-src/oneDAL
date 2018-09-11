/* file: truncated_gaussian_initializer_types.h */
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
