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
 * @ingroup dbscan_compute
 * @{
 */
package com.intel.daal.algorithms.dbscan;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__RESULTID"></a>
 * @brief Available identifiers of the results of the DBSCAN algorithm
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

    private static final int assignmentsValue      = 0;
    private static final int nClustersValue        = 1;
    private static final int coreIndicesValue      = 2;
    private static final int coreObservationsValue = 3;

    public static final ResultId assignments      = new ResultId(assignmentsValue);         /*!< Assignments of observations to clusters */
    public static final ResultId nClusters        = new ResultId(nClustersValue);           /*!< Number of clusters */
    public static final ResultId coreIndices      = new ResultId(coreIndicesValue);         /*!< Indices of core observations */
    public static final ResultId coreObservations = new ResultId(coreObservationsValue);    /*!< Core observations */

}
/** @} */
