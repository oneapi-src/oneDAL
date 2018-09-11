/* file: distribution.h */
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
//  Implementation of distributions
//--
*/

#ifndef __DISTRIBUTIONS_H__
#define __DISTRIBUTIONS_H__

#include "algorithms/distributions/distribution_types.h"

namespace daal
{
namespace algorithms
{
namespace distributions
{
/**
 * @ingroup distributions
 * @{
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTIONS__BATCHBASE"></a>
 *  \brief Class representing distributions
 */
class DAAL_EXPORT BatchBase : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::distributions::Input         InputType;
    typedef algorithms::distributions::ParameterBase ParameterType;
    typedef algorithms::distributions::Result        ResultType;

    InputType  input;  /*!< Input of the distribution */
    virtual ~BatchBase() {}
};

} // namespace interface1
using interface1::BatchBase;
/** @} */
} // namespace distributions
} // namespace algorithms
} // namespace daal
#endif
