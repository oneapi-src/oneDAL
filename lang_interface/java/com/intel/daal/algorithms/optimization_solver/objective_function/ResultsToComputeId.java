/* file: ResultsToComputeId.java */
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

/**
 * @ingroup objective_function
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.objective_function;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__RESULTSTOCOMPUTEID"></a>
 * @brief Available computation flag identifiers for the objective funtion result
 */
public final class ResultsToComputeId {

    public static final long gradient = 0x0000000000000001L; /*!< Objective function gradient compute flag */
    public static final long value    = 0x0000000000000002L; /*!< Objective function value compute flag */
    public static final long hessian  = 0x0000000000000004L; /*!< Objective function hessian compute flag */
}
/** @} */
