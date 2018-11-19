/* file: normal_types.h */
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
//  Implementation of normal distribution.
//--
*/

#ifndef __NORMAL__TYPES_H__
#define __NORMAL__TYPES_H__

#include "algorithms/distributions/distribution_types.h"

namespace daal
{
namespace algorithms
{
namespace distributions
{
/**
 * @defgroup distributions_normal Normal Distribution
 * \copydoc daal::algorithms::distributions::normal
 * @ingroup distributions
 * @{
 */
/**
 * \brief Contains classes for normal distribution
 */
namespace normal
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__DISTRIBUTIONS__NORMAL__METHOD"></a>
 * Available methods to compute normal distribution
 */
enum Method
{
    icdf         = 0,     /*!< Default: Inverse cumulative distribution function method. */
    defaultDense = 0    /*!< Default: performance-oriented method. */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTIONS__NORMAL__PARAMETER"></a>
 * \brief Normal distribution parameters
 */
template<typename algorithmFPType>
class DAAL_EXPORT Parameter: public distributions::ParameterBase
{
public:
    /**
     *  Main constructor
     *  \param[in] _a     Mean
     *  \param[in] _sigma Standard deviation
     */
    Parameter(algorithmFPType _a = 0.0, algorithmFPType _sigma = 1.0): a(_a), sigma(_sigma) {}

    algorithmFPType a;        /*!< Mean */
    algorithmFPType sigma;    /*!< Standard deviation */

     /**
     * Check the correctness of the %Parameter object
     */
    services::Status check() const DAAL_C11_OVERRIDE;
};

} // namespace interface1
using interface1::Parameter;

} // namespace normal
/** @} */
} // namespace distributions
} // namespace algorithms
} // namespace daal

#endif
