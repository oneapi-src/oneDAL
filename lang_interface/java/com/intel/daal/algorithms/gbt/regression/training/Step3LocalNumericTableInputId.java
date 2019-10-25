/* file: Step3LocalNumericTableInputId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__STEP3LOCALNUMERICTABLEINPUTID"></a>
 * @brief Available identifiers of input data numeric table objects for model-based training in the third step
 *        of the distributed processing mode
 */
public final class Step3LocalNumericTableInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public Step3LocalNumericTableInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step3BinnedDataValue = 0;
    private static final int step3BinSizesValue = 1;
    private static final int step3InputTreeStructureValue = 2;
    private static final int step3InputTreeOrderValue = 3;
    private static final int step3OptCoeffsValue = 4;

    public static final Step3LocalNumericTableInputId step3BinnedData = new Step3LocalNumericTableInputId(step3BinnedDataValue);
        /*!<  */
    public static final Step3LocalNumericTableInputId step3BinSizes = new Step3LocalNumericTableInputId(step3BinSizesValue);
        /*!<  */
    public static final Step3LocalNumericTableInputId step3InputTreeStructure = new Step3LocalNumericTableInputId(step3InputTreeStructureValue);
        /*!<  */
    public static final Step3LocalNumericTableInputId step3InputTreeOrder = new Step3LocalNumericTableInputId(step3InputTreeOrderValue);
        /*!<  */
    public static final Step3LocalNumericTableInputId step3OptCoeffs = new Step3LocalNumericTableInputId(step3OptCoeffsValue);
        /*!<  */
}
/** @} */
