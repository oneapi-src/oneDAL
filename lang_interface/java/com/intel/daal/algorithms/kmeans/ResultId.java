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
 * @ingroup kmeans_compute
 * @{
 */
package com.intel.daal.algorithms.kmeans;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__RESULTID"></a>
 * @brief Available identifiers of the results of the K-Means algorithm
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

    private static final int centroidsValue    = 0;
    private static final int assignmentsValue  = 1;
    private static final int goalFunctionValue = 2;
    private static final int objectiveFunctionValue = 2;
    private static final int nIterationsValue  = 3;

    public static final ResultId centroids    = new ResultId(centroidsValue);    /*!< Centroids */
    public static final ResultId assignments  = new ResultId(assignmentsValue);  /*!< Assignment of observations to clusters */
    public static final ResultId objectiveFunction = new ResultId(objectiveFunctionValue); /*!< Value of the objective function */
    public static final ResultId goalFunction = new ResultId(goalFunctionValue); /*!< Value of the objective function @DAAL_DEPRECATED */
    public static final ResultId nIterations  = new ResultId(nIterationsValue);  /*!< Number of executed iterations */

}
/** @} */
