/* file: Step1LocalNumericTableInputId.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
