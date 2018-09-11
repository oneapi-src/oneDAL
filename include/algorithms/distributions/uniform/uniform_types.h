/* file: uniform_types.h */
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
//  Implementation of uniform distribution.
//--
*/

#ifndef __UNIFORM__TYPES_H__
#define __UNIFORM__TYPES_H__

#include "algorithms/distributions/distribution_types.h"

namespace daal
{
namespace algorithms
{
namespace distributions
{
/**
 * @defgroup distributions_uniform Uniform Distribution
 * \copydoc daal::algorithms::distributions::uniform
 * @ingroup distributions
 * @{
 */
/**
 * \brief Contains classes for uniform distribution
 */
namespace uniform
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__DISTRIBUTIONS__UNIFORM__METHOD"></a>
 * Available methods to compute uniform distribution
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
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTIONS__UNIFORM__PARAMETER"></a>
 * \brief Uniform distribution parameters
 */
template<typename algorithmFPType>
class DAAL_EXPORT Parameter: public distributions::ParameterBase
{
public:
    /**
     *  Main constructor
     *  \param[in] _a    Left bound a
     *  \param[in] _b    Right bound b
     */
    Parameter(algorithmFPType _a = 0.0, algorithmFPType _b = 1.0): a(_a), b(_b) {}

    algorithmFPType a;    /*!< Left bound a */
    algorithmFPType b;    /*!< Right bound b */

    /**
     * Check the correctness of the %Parameter object
     */
    services::Status check() const DAAL_C11_OVERRIDE;
};

} // namespace interface1
using interface1::Parameter;

} // namespace uniform
/** @} */
} // namespace distributions
} // namespace algorithms
} // namespace daal

#endif
