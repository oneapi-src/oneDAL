/* file: PruningId.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
* @defgroup decision_tree Decision tree
 * @ingroup training_and_prediction
 * @{
 */

package com.intel.daal.algorithms.decision_tree;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__TRAINING__PRUNINGID"></a>
 * @brief Pruning method for Decision tree algorithm
 */
public final class PruningId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the pruning object identifier using the provided value
     * @param value     Value corresponding to the pruning object identifier
     */
    public PruningId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the pruning object identifier
     * @return Value corresponding to the pruning object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int noneId                = 0;
    private static final int reducedErrorPruningId = 1;

    public static final PruningId none                = new PruningId(noneId);                /*!< Do not prune */
    public static final PruningId reducedErrorPruning = new PruningId(reducedErrorPruningId); /*!< Reduced error pruning */
}
/** @} */
