/* file: InitStep2MasterCollectionInputId.java */
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
package com.intel.daal.algorithms.gbt.regression.init;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__STEP2MASTERCOLLECTIONINPUTID"></a>
 * @brief Available identifiers of input numeric table objects for the DBSCAN algorithm in the eleventh step
 *        of the distributed processing mode
 */
public final class InitStep2MasterCollectionInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public InitStep2MasterCollectionInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step2MeanDependentVariableValue = 0;
    private static final int step2NumberOfRowsValue = 1;
    private static final int step2BinBordersValue = 2;
    private static final int step2BinSizesValue = 3;

    public static final InitStep2MasterCollectionInputId step2MeanDependentVariable = new InitStep2MasterCollectionInputId(step2MeanDependentVariableValue);
        /*!<  */
    public static final InitStep2MasterCollectionInputId step2NumberOfRows = new InitStep2MasterCollectionInputId(step2NumberOfRowsValue);
        /*!<  */
    public static final InitStep2MasterCollectionInputId step2BinBorders = new InitStep2MasterCollectionInputId(step2BinBordersValue);
        /*!<  */
    public static final InitStep2MasterCollectionInputId step2BinSizes = new InitStep2MasterCollectionInputId(step2BinSizesValue);
        /*!<  */
}
/** @} */
