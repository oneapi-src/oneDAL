/* file: precomputed_types.h */
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
//  Implementation of the objective function with precomputed characteristics
//  interface.
//--
*/

#ifndef __PRECOMPUTED_TYPES_H__
#define __PRECOMPUTED_TYPES_H__

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
/**
 * @defgroup precomputed Objective function with precomputed characteristics
 * \copydoc daal::algorithms::optimization_solver::precomputed
 * @ingroup objective_function
 * @{
 */
/**
 * \brief Contains classes for the Objective function with precomputed characteristics
 */
namespace precomputed
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__PRECOMPUTED__METHOD"></a>
 * Available methods for computing results of Objective function with precomputed characteristics
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method. */
};
}
/** @} */
}
}
}
#endif
