/* file: InitStep1LocalNumericTableInputId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__STEP11LOCALNUMERICTABLEINPUTID"></a>
 * @brief Available identifiers of input numeric table objects for the DBSCAN algorithm in the eleventh step
 *        of the distributed processing mode
 */
public final class InitStep1LocalNumericTableInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public InitStep1LocalNumericTableInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step1LocalDataValue = 0;
    private static final int step1LocalDependentVariablesValue = 1;

    public static final InitStep1LocalNumericTableInputId step1LocalData = new InitStep1LocalNumericTableInputId(step1LocalDataValue);
        /*!<  */
    public static final InitStep1LocalNumericTableInputId step1LocalDependentVariables = new InitStep1LocalNumericTableInputId(step1LocalDependentVariablesValue);
       /*!<  */
}
/** @} */
