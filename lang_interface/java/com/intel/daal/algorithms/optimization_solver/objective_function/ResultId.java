/* file: ResultId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__RESULTID"></a>
 * @brief Available result identifiers for the objective funtion algorithm
 */
public final class ResultId {
    private int _value;

    /**
     * Constructs the result object identifier using the provided value
     * @param value     Value corresponding to the result object identifier
     */
    public ResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result object identifier
     * @return Value corresponding to the result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int gradientIdxId = 0;
    private static final int valueIdxId    = 1;
    private static final int hessianIdxId  = 2;

    public static final ResultId gradientIdx  = new ResultId(gradientIdxId); /*!< Objective function gradient index */
    public static final ResultId valueIdx  = new ResultId(valueIdxId);       /*!< Objective function value index */
    public static final ResultId hessianIdx  = new ResultId(hessianIdxId);   /*!< Objective function hessian index */
}
/** @} */
