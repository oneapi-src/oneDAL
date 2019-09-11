/* file: MultiClassConfusionMatrixInput.java */
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
 * @ingroup quality_metric_multiclass
 * @{
 */
package com.intel.daal.algorithms.classifier.quality_metric.multi_class_confusion_matrix;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__MULTI_CLASS_CONFUSION_MATRIX__MULTICLASSCONFUSIONMATRIXINPUT"></a>
 * @brief  Class for the input objects of the multi-class confusion matrix algorithm
 */
public class MultiClassConfusionMatrixInput extends com.intel.daal.algorithms.quality_metric.QualityMetricInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public MultiClassConfusionMatrixInput(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Sets the input object for the multi-class confusion matrix algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(MultiClassConfusionMatrixInputId id, NumericTable val) {
        if (id != MultiClassConfusionMatrixInputId.predictedLabels
                && id != MultiClassConfusionMatrixInputId.groundTruthLabels) {
            throw new IllegalArgumentException("id unsupported");
        }

        cSetInputTable(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns the input object of the confusion matrix of the classification algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(MultiClassConfusionMatrixInputId id) {
        if (id == MultiClassConfusionMatrixInputId.predictedLabels
                || id == MultiClassConfusionMatrixInputId.groundTruthLabels) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetInputTable(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInputTable(long inputAddr, int id, long ntAddr);

    private native long cGetInputTable(long cInput, int id);
}
/** @} */
