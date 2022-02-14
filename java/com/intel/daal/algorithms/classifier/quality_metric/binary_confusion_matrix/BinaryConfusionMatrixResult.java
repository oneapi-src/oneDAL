/* file: BinaryConfusionMatrixResult.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
 * @ingroup quality_metric_binary
 * @{
 */
package com.intel.daal.algorithms.classifier.quality_metric.binary_confusion_matrix;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__BINARY_CONFUSION_MATRIX__BINARYCONFUSIONMATRIXRESULT"></a>
 * @brief  Class for the results of the binary confusion matrix algorithm
 */
public class BinaryConfusionMatrixResult extends com.intel.daal.algorithms.quality_metric.QualityMetricResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public BinaryConfusionMatrixResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Constructs the result of the classification algorithms
     * @param context   Context to manage the result of the classification algorithms
     */
    public BinaryConfusionMatrixResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    /**
     * Sets the result of the binary confusion matrix algorithm
     * @param id    Identifier of the result
     * @param val   Value that corresponds to the given identifier
     */
    public void set(BinaryConfusionMatrixResultId id, NumericTable val) {
        if (id != BinaryConfusionMatrixResultId.confusionMatrix || id != BinaryConfusionMatrixResultId.binaryMetrics) {
            throw new IllegalArgumentException("id unsupported");
        }

        cSetResultTable(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns the quality metric of the classification algorithm
     * @param id Identifier of the result
     * @return   Result that corresponds to the given identifier
     */
    public NumericTable get(BinaryConfusionMatrixResultId id) {
        if (id == BinaryConfusionMatrixResultId.confusionMatrix || id == BinaryConfusionMatrixResultId.binaryMetrics) {
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
