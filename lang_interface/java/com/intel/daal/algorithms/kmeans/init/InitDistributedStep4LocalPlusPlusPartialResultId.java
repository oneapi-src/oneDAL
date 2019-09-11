/* file: InitDistributedStep4LocalPlusPlusPartialResultId.java */
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
 * @ingroup kmeans_init_distributed
 * @{
 */
package com.intel.daal.algorithms.kmeans.init;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP4LOCALPLUSPLUSPARTIALRESULTID"></a>
 * @brief Available identifiers of partial results of computing initial clusters for the K-Means algorithm in the distributed processing mode
 *        used with plusPlus and parallelPlus methods only on the 4th step on a local node.
 */
public final class InitDistributedStep4LocalPlusPlusPartialResultId {
    private int _value;

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public InitDistributedStep4LocalPlusPlusPartialResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int outputOfStep4Value = 0;

    /** NumericTable with the new centroids calculated on step 4 on the local node */
    public static final InitDistributedStep4LocalPlusPlusPartialResultId outputOfStep4 = new InitDistributedStep4LocalPlusPlusPartialResultId(
            outputOfStep4Value);
}
/** @} */
