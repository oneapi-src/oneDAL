/* file: Step5LocalNumericTableInputId.java */
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
 * @ingroup gbt_compute
 * @{
 */
package com.intel.daal.algorithms.gbt.regression.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__STEP5LOCALNUMERICTABLEINPUTID"></a>
 * @brief Available identifiers of input data numeric table objects for model-based training in the fifth step
 *        of the distributed processing mode
 */
public final class Step5LocalNumericTableInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public Step5LocalNumericTableInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step5BinnedDataValue = 0;
    private static final int step5TransposedBinnedDataValue = 1;
    private static final int step5BinSizesValue = 2;
    private static final int step5InputTreeStructureValue = 3;
    private static final int step5InputTreeOrderValue = 4;

    public static final Step5LocalNumericTableInputId step5BinnedData = new Step5LocalNumericTableInputId(step5BinnedDataValue);
        /*!<  */
    public static final Step5LocalNumericTableInputId step5TransposedBinnedData = new Step5LocalNumericTableInputId(step5TransposedBinnedDataValue);
        /*!<  */
    public static final Step5LocalNumericTableInputId step5BinSizes = new Step5LocalNumericTableInputId(step5BinSizesValue);
        /*!<  */
    public static final Step5LocalNumericTableInputId step5InputTreeStructure = new Step5LocalNumericTableInputId(step5InputTreeStructureValue);
        /*!<  */
    public static final Step5LocalNumericTableInputId step5InputTreeOrder = new Step5LocalNumericTableInputId(step5InputTreeOrderValue);
        /*!<  */
}
/** @} */
