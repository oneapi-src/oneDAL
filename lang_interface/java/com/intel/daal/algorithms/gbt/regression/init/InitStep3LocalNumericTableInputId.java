/* file: InitStep3LocalNumericTableInputId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__STEp31LOCALNUMERICTABLEINPUTID"></a>
 * @brief Available identifiers of input numeric table objects for the DBSCAN algorithm in the eleventh step
 *        of the distributed processing mode
 */
public final class InitStep3LocalNumericTableInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public InitStep3LocalNumericTableInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step3MergedBinBordersValue = 0;
    private static final int step3BinQuantitiesValue = 1;
    private static final int step3LocalDataValue = 2;
    private static final int step3InitialResponseValue = 3;

    public static final InitStep3LocalNumericTableInputId step3MergedBinBorders = new InitStep3LocalNumericTableInputId(step3MergedBinBordersValue);
        /*!<  */
    public static final InitStep3LocalNumericTableInputId step3BinQuantities = new InitStep3LocalNumericTableInputId(step3BinQuantitiesValue);
       /*!<  */
    public static final InitStep3LocalNumericTableInputId step3LocalData = new InitStep3LocalNumericTableInputId(step3LocalDataValue);
    /*!<  */
    public static final InitStep3LocalNumericTableInputId step3InitialResponse = new InitStep3LocalNumericTableInputId(step3InitialResponseValue);
    /*!<  */
}
/** @} */
