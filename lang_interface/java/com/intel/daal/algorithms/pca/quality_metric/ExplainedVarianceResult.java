/* file: ExplainedVarianceResult.java */
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
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.DataCollection;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__QUALITY_METRIC__EXPLAINEDVARIANCERESULT"></a>
 * @brief  Class for the the result of PCA quality metrics algorithm
 */
public class ExplainedVarianceResult extends com.intel.daal.algorithms.quality_metric.QualityMetricResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public ExplainedVarianceResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Constructs the result of the quality metric algorithm
     * @param context   Context to manage the result of the quality metric algorithm
     */
    public ExplainedVarianceResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    /**
     * Sets the result of PCA quality metrics
     * @param id    Identifier of the result
     * @param val   Value that corresponds to the given identifier
     */
    public void set(ExplainedVarianceResultId id, NumericTable val) {
        if (id == ExplainedVarianceResultId.explainedVariances ||
            id == ExplainedVarianceResultId.explainedVariancesRatios ||
            id == ExplainedVarianceResultId.noiseVariance) {
            cSetResultTable(cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns the result of PCA quality metrics
     * @param id Identifier of the result
     * @return   Result that corresponds to the given identifier
     */
    public NumericTable get(ExplainedVarianceResultId id) {
        if (id == ExplainedVarianceResultId.explainedVariances ||
            id == ExplainedVarianceResultId.explainedVariancesRatios ||
            id == ExplainedVarianceResultId.noiseVariance) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetResultTable(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native long cNewResult();
    private native void cSetResultTable(long inputAddr, int id, long ntAddr);
    private native long cGetResultTable(long cResult, int id);
}
/** @} */
