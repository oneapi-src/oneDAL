/* file: MultiClassConfusionMatrixResult.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__MULTI_CLASS_CONFUSION_MATRIX__MULTICLASSCONFUSIONMATRIXRESULT"></a>
 * @brief  Class for the results of the multi-class confusion matrix algorithm
 */
public class MultiClassConfusionMatrixResult extends com.intel.daal.algorithms.quality_metric.QualityMetricResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public MultiClassConfusionMatrixResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Constructs the result of the multi-class confusion matrix algorithm
     * @param context   Context to manage the result of the multi-class confusion matrix algorithm
     */
    public MultiClassConfusionMatrixResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    /**
     * Sets the result of the training stage of the classification algorithm
     * @param id    Identifier of the result
     * @param val   Value that corresponds to the given identifier
     */
    public void set(MultiClassConfusionMatrixResultId id, NumericTable val) {
        if (id != MultiClassConfusionMatrixResultId.confusionMatrix
                && id != MultiClassConfusionMatrixResultId.multiClassMetrics) {
            throw new IllegalArgumentException("id unsupported");
        }

        cSetResultTable(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns the confusion matrix of the multi-class classification algorithm
     * @param id Identifier of the result
     * @return   Result that corresponds to the given identifier
     */
    public NumericTable get(MultiClassConfusionMatrixResultId id) {
        if (id == MultiClassConfusionMatrixResultId.confusionMatrix
                || id == MultiClassConfusionMatrixResultId.multiClassMetrics) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetResultTable(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetResultTable(long inputAddr, int id, long ntAddr);

    private native long cGetResultTable(long cResult, int id);

    private native long cNewResult();

}
/** @} */
