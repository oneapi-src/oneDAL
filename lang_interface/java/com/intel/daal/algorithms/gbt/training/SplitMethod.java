/* file: SplitMethod.java */
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
 * @ingroup gbt
 */
/**
 * @brief Contains classes of the gradient boosted trees algorithm training
 */
package com.intel.daal.algorithms.gbt.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__TRAINING__SPLITMETHOD"></a>
 * @brief Split finding method in gradient boosted trees algorithm
 */
public final class SplitMethod {
    private int _value;

    /**
     * Constructs the split method identifier using the provided value
     * @param value     Value corresponding to the split method identifier
     */
    public SplitMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the split method identifier
     * @return Value corresponding to the split method identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int exactId        = 0;
    private static final int inexactId      = 1;
    private static final int defaultSplitId = inexactId;

    public static final SplitMethod exact        = new SplitMethod(exactId);        /*!< Exact greedy method */
    public static final SplitMethod inexact      = new SplitMethod(inexactId);      /*!< Inexact method for splits finding: bucket continuous features to discrete bins */
    public static final SplitMethod defaultSplit = new SplitMethod(defaultSplitId); /*!< Default split finding method */
}
/** @} */
