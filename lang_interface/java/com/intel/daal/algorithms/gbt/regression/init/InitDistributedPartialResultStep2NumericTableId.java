/* file: InitDistributedPartialResultStep2NumericTableId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__InitDistributedPartialResultStep2NumericTableId"></a>
 * @brief Available identifiers of partial results of the DBSCAN training algorithm obtained
 * in the sixth step of the distributed processing mode
 */
public final class InitDistributedPartialResultStep2NumericTableId {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public InitDistributedPartialResultStep2NumericTableId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step2MergedBinBordersValue = 0;
    private static final int step2BinQuantitiesValue = 1;
    private static final int step2InitialResponseValue = 2;

    public static final InitDistributedPartialResultStep2NumericTableId step2MergedBinBorders = new InitDistributedPartialResultStep2NumericTableId(step2MergedBinBordersValue);
        /*!<  */
    public static final InitDistributedPartialResultStep2NumericTableId step2BinQuantities = new InitDistributedPartialResultStep2NumericTableId(step2BinQuantitiesValue);
        /*!<  */
    public static final InitDistributedPartialResultStep2NumericTableId step2InitialResponse = new InitDistributedPartialResultStep2NumericTableId(step2InitialResponseValue);
        /*!<  */
}
/** @} */
