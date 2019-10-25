/* file: InitDistributedPartialResultStep1Id.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__InitDistributedPartialResultStep1Id"></a>
 * @brief Available identifiers of partial results of the DBSCAN training algorithm obtained
 * in the sixth step of the distributed processing mode
 */
public final class InitDistributedPartialResultStep1Id {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public InitDistributedPartialResultStep1Id(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step1BinBordersValue = 0;
    private static final int step1BinSizesValue = 1;
    private static final int step1MeanDependentVariableValue = 2;
    private static final int step1NumberOfRowsValue = 3;

    public static final InitDistributedPartialResultStep1Id step1BinBorders = new InitDistributedPartialResultStep1Id(step1BinBordersValue);
        /*!<  */
    public static final InitDistributedPartialResultStep1Id step1BinSizes = new InitDistributedPartialResultStep1Id(step1BinSizesValue);
        /*!<  */
    public static final InitDistributedPartialResultStep1Id step1MeanDependentVariable = new InitDistributedPartialResultStep1Id(step1MeanDependentVariableValue);
        /*!<  */
    public static final InitDistributedPartialResultStep1Id step1NumberOfRows = new InitDistributedPartialResultStep1Id(step1NumberOfRowsValue);
        /*!<  */
}
/** @} */
