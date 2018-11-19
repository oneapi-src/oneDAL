/* file: VariableImportanceModeId.java */
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
* @defgroup decision_forest Decision forest
 * @ingroup training_and_prediction
 * @{
 */

package com.intel.daal.algorithms.decision_forest;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__TRAINING__VARIABLEIMPORTANCEMODEID"></a>
 * @brief Variable importance computation mode
 */
public final class VariableImportanceModeId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the variable importance computation mode object identifier using the provided value
     * @param value     Value corresponding to the variable importance computation mode object identifier
     */
    public VariableImportanceModeId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the variable importance computation mode object identifier
     * @return Value corresponding to the variable importance computation mode object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int noneId = 0;
    private static final int MDIId = 1;
    private static final int MDA_RawId = 2;
    private static final int MDA_ScaledId = 3;

    public static final VariableImportanceModeId none = new VariableImportanceModeId(noneId);
        /*!< Do not compute */
    public static final VariableImportanceModeId MDI = new VariableImportanceModeId(MDIId);
        /*!< Mean Decrease Impurity. Computed as the sum of weighted impurity decreases for all nodes where the variable is used,
             averaged over all trees in the forest */
    public static final VariableImportanceModeId MDA_Raw = new VariableImportanceModeId(MDA_RawId);
        /*!< Mean Decrease Accuracy (permutation importance).
             For each tree, the prediction error on the out-of-bag portion of the data is computed
             (error rate for classification, MSE for regression).
             The same is done after permuting each predictor variable.
             The difference between the two are then averaged over all trees. */
    public static final VariableImportanceModeId MDA_Scaled = new VariableImportanceModeId(MDA_ScaledId);
        /*!< Mean Decrease Accuracy (permutation importance).
             This is MDA_Raw value normalized by its standard deviation. */
}
/** @} */
