/* file: bernoulli_types.h */
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
//  Implementation of bernoulli distribution.
//--
*/

#ifndef __BERNOULLI__TYPES_H__
#define __BERNOULLI__TYPES_H__

#include "algorithms/distributions/distribution_types.h"

namespace daal
{
namespace algorithms
{
namespace distributions
{
/**
 * @defgroup distributions_bernoulli Bernoulli Distribution
 * \copydoc daal::algorithms::distributions::bernoulli
 * @ingroup distributions
 * @{
 */
/**
 * \brief Contains classes for bernoulli distribution
 */
namespace bernoulli
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__DISTRIBUTIONS__BERNOULLI__METHOD"></a>
 * Available methods to compute bernoulli distribution
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
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTIONS__BERNOULLI__PARAMETER"></a>
 * \brief Bernoulli distribution parameters
 */
template<typename algorithmFPType>
class DAAL_EXPORT Parameter: public distributions::ParameterBase
{
public:
    /**
     *  Main constructor
     *  \param[in] _p    Success probability of a trial, value from [0.0; 1.0]
     */
    Parameter(algorithmFPType _p): p(_p) {}

    algorithmFPType p;    /*!< Success probability of a trial, value from [0.0; 1.0] */

    /**
     * Check the correctness of the %Parameter object
     */
    services::Status check() const DAAL_C11_OVERRIDE;
};

} // namespace interface1
using interface1::Parameter;

} // namespace bernoulli
/** @} */
} // namespace distributions
} // namespace algorithms
} // namespace daal

#endif
