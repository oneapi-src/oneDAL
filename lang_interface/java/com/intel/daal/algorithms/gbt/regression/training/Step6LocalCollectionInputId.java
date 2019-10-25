/* file: Step6LocalCollectionInputId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__STEP6LOCALCOLLECTIONINPUTID"></a>
 * @brief Available identifiers of input data collection table objects for model-based training in the sixth step
 *        of the distributed processing mode
 */
public final class Step6LocalCollectionInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public Step6LocalCollectionInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step6BinValuesValue = 1;
    private static final int step6FinalizedTreesValue = 2;

    public static final Step6LocalCollectionInputId step6BinValues = new Step6LocalCollectionInputId(step6BinValuesValue);
        /*!<  */
    public static final Step6LocalCollectionInputId step6FinalizedTrees = new Step6LocalCollectionInputId(step6FinalizedTreesValue);
        /*!<  */
}
/** @} */
