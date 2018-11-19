/* file: InitPartialResultId.java */
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
 * @ingroup kmeans_init
 * @{
 */
package com.intel.daal.algorithms.kmeans.init;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITPARTIALRESULTID"></a>
 * @brief Available identifiers of partial results of computing initial clusters for the K-Means algorithm
 */
public final class InitPartialResultId {
    private int _value;

    /**
     * Constructs the initialization partial result object identifier using the provided value
     * @param value     Value corresponding to the initialization partial result object identifier
     */
    public InitPartialResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the initialization partial result object identifier
     * @return Value corresponding to the initialization partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int PartialClusters       = 0;
    private static final int PartialCentroids      = 0;
    private static final int PartialClustersNumber = 1;

    /** Sum of observations */
    public static final InitPartialResultId partialCentroids      = new InitPartialResultId(PartialCentroids);
    /** Sum of observations @DAAL_DEPRECATED */
    public static final InitPartialResultId partialClusters       = new InitPartialResultId(PartialClusters);
    /** Number of assigned observations @DAAL_DEPRECATED */
    public static final InitPartialResultId partialClustersNumber = new InitPartialResultId(PartialClustersNumber);
}
/** @} */
