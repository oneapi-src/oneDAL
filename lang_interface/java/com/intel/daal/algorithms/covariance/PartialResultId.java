/* file: PartialResultId.java */
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
 * @ingroup covariance
 * @{
 */
package com.intel.daal.algorithms.covariance;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__PARTIALRESULTID"></a>
 * @brief Available identifiers of partial results of the correlation or variance-covariance matrix algorithm
 */
public final class PartialResultId {
    private int _value;

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public PartialResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int nObservationsValue = 0;
    private static final int crossProductValue  = 1;
    private static final int sumValue           = 2;

    /** Number of observations processed so far */
    public static final PartialResultId nObservations = new PartialResultId(nObservationsValue);
    /** Cross-product matrix computed so far */
    public static final PartialResultId crossProduct  = new PartialResultId(crossProductValue);
    /** Vector of sums computed so far */
    public static final PartialResultId sum           = new PartialResultId(sumValue);
}
/** @} */
