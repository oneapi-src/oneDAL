/* file: optimization_solver_batch.h */
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
//  Implementation of Optimization solver interface interface.
//--
*/

#ifndef __OPTIMIZATION_SOLVER_BATCH_H__
#define __OPTIMIZATION_SOLVER_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
/**
* \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
*/
namespace interface1
{
/**
 * @addtogroup optimization_solver
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__BATCHIFACE"></a>
 * \brief Interface for computing the Optimization solver in the %batch processing mode.
 */
class DAAL_EXPORT BatchIface : public daal::algorithms::Analysis<batch>
{
public:
    BatchIface() {}
    virtual ~BatchIface() {}
};
/** @} */
} // namespace interface1
using interface1::BatchIface;

} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
