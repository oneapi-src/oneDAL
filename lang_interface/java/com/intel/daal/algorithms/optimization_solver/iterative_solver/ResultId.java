/* file: ResultId.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
 * @ingroup iterative_solver
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.iterative_solver;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__RESULTID"></a>
 * @brief Available result identifiers for the iterative solver algorithm
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

    private static final int minimumId = 0;
    private static final int nIterationsId = 1;

    public static final ResultId minimum  = new ResultId(minimumId); /*!< Table to store the obtained minimum value */
    public static final ResultId nIterations = new ResultId(nIterationsId); /*!< Table containing the number of executed iterations */
}
/** @} */
