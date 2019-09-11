/* file: ExplainedVarianceInput.java */
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
 * @ingroup pca_quality_metric_explained_variance
 * @{
 */
package com.intel.daal.algorithms.pca.quality_metric;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__QUALITY_METRIC__EXPLAINEDVARIANCEINPUT"></a>
 * @brief  Class for the input objects of the algorithm
 */
public class ExplainedVarianceInput extends com.intel.daal.algorithms.quality_metric.QualityMetricInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public ExplainedVarianceInput(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Sets the input object for PCA quality metric
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(ExplainedVarianceInputId id, NumericTable val) {
        if (id != ExplainedVarianceInputId.eigenValues) {
            throw new IllegalArgumentException("id unsupported");
        }

        cSetInputTable(this.cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns the input object for PCA quality metric
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(ExplainedVarianceInputId id) {
        if (id == ExplainedVarianceInputId.eigenValues) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetInputTable(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInputTable(long inputAddr, int id, long ntAddr);
    private native long cGetInputTable(long cInput, int id);
}
/** @} */
