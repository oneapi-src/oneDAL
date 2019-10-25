/* file: Step1LocalNumericTableInputId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__STEP1LOCALNUMERICTABLEINPUTID"></a>
 * @brief Available identifiers of input data numeric table objects for model-based training in the first step
 *        of the distributed processing mode
 */
public final class Step1LocalNumericTableInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public Step1LocalNumericTableInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step1BinnedDataValue = 0;
    private static final int step1DependentVariableValue = 1;
    private static final int step1InputResponseValue = 2;
    private static final int step1InputTreeStructureValue = 3;
    private static final int step1InputTreeOrderValue = 4;

    public static final Step1LocalNumericTableInputId step1BinnedData = new Step1LocalNumericTableInputId(step1BinnedDataValue);
        /*!<  */
    public static final Step1LocalNumericTableInputId step1DependentVariable = new Step1LocalNumericTableInputId(step1DependentVariableValue);
        /*!<  */
    public static final Step1LocalNumericTableInputId step1InputResponse = new Step1LocalNumericTableInputId(step1InputResponseValue);
        /*!<  */
    public static final Step1LocalNumericTableInputId step1InputTreeStructure = new Step1LocalNumericTableInputId(step1InputTreeStructureValue);
        /*!<  */
    public static final Step1LocalNumericTableInputId step1InputTreeOrder = new Step1LocalNumericTableInputId(step1InputTreeOrderValue);
        /*!<  */
}
/** @} */
