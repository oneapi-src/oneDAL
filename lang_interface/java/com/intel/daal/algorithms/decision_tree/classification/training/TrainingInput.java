/* file: TrainingInput.java */
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
 * @defgroup decision_tree_classification_training Training
 * @brief Contains classes for training based on decision tree classification models
 * @ingroup decision_tree_classification
 * @{
 */
package com.intel.daal.algorithms.decision_tree.classification.training;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__CLASSIFICATION__TRAINING__TRAININGINPUT"></a>
 * @brief  %Input objects for the decision tree classification algorithm
 */
public class TrainingInput extends com.intel.daal.algorithms.classifier.training.TrainingInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public TrainingInput(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Sets the NumericTable input object for the decision tree classification model-based training algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(TrainingInputId id, NumericTable val) {
        if (id != TrainingInputId.dataForPruning && id != TrainingInputId.labelsForPruning) {
            throw new IllegalArgumentException("id unsupported");
        }

        cSetInputTable(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns the NumericTable input object for the decision tree classification model-based training algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(TrainingInputId id) {
        if (id == TrainingInputId.dataForPruning || id == TrainingInputId.labelsForPruning) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetInputTable(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInputTable(long inputAddr, int id, long ntAddr);

    private native long cGetInputTable(long inputAddr, int id);
}
/** @} */
