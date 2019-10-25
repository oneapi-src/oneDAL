/* file: InitDistributedPartialResultStep3Id.java */
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
 * @ingroup dbscan_distributed
 * @{
 */
package com.intel.daal.algorithms.gbt.regression.init;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__InitDistributedPartialResultStep3Id"></a>
 * @brief Available identifiers of partial results of the DBSCAN training algorithm obtained
 * in the sixth step of the distributed processing mode
 */
public final class InitDistributedPartialResultStep3Id {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public InitDistributedPartialResultStep3Id(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step3BinnedDataValue = 0;
    private static final int step3TransposedBinnedDataValue = 1;
    private static final int step3ResponseValue = 2;
    private static final int step3TreeOrderValue = 3;

    public static final InitDistributedPartialResultStep3Id step3BinnedData = new InitDistributedPartialResultStep3Id(step3BinnedDataValue);
        /*!<  */
    public static final InitDistributedPartialResultStep3Id step3TransposedBinnedData = new InitDistributedPartialResultStep3Id(step3TransposedBinnedDataValue);
        /*!<  */
    public static final InitDistributedPartialResultStep3Id step3Response = new InitDistributedPartialResultStep3Id(step3ResponseValue);
        /*!<  */
    public static final InitDistributedPartialResultStep3Id step3TreeOrder = new InitDistributedPartialResultStep3Id(step3TreeOrderValue);
        /*!<  */
}
/** @} */
