/* file: ExplainedVarianceResult.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
