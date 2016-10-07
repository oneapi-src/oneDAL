/* file: GroupOfBetasResult.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

package com.intel.daal.algorithms.linear_regression.quality_metric;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__GROUPOFBETASRESULT"></a>
 * @brief  Class for the the result of linear regression quality metrics algorithm
 */
public class GroupOfBetasResult extends com.intel.daal.algorithms.quality_metric.QualityMetricResult {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public GroupOfBetasResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    public GroupOfBetasResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    /**
     * Sets the result of linear regression quality metrics
     * @param id    Identifier of the result
     * @param val   Value that corresponds to the given identifier
     */
    public void set(GroupOfBetasResultId id, NumericTable val) {
        if (id == GroupOfBetasResultId.expectedMeans ||
            id == GroupOfBetasResultId.expectedVariance ||
            id == GroupOfBetasResultId.regSS ||
            id == GroupOfBetasResultId.resSS ||
            id == GroupOfBetasResultId.tSS ||
            id == GroupOfBetasResultId.determinationCoeff ||
            id == GroupOfBetasResultId.fStatistics)
            cSetResultTable(cObject, id.getValue(), val.getCObject());
        else
            throw new IllegalArgumentException("id unsupported");
    }

    /**
     * Returns the result of linear regression quality metrics
     * @param id Identifier of the result
     * @return   Result that corresponds to the given identifier
     */
    public NumericTable get(GroupOfBetasResultId id) {
        if (id == GroupOfBetasResultId.expectedMeans ||
            id == GroupOfBetasResultId.expectedVariance ||
            id == GroupOfBetasResultId.regSS ||
            id == GroupOfBetasResultId.resSS ||
            id == GroupOfBetasResultId.tSS ||
            id == GroupOfBetasResultId.determinationCoeff ||
            id == GroupOfBetasResultId.fStatistics)
            return new HomogenNumericTable(getContext(), cGetResultTable(cObject, id.getValue()));
        throw new IllegalArgumentException("id unsupported");
    }

    private native void cSetResultTable(long inputAddr, int id, long ntAddr);
    private native long cGetResultTable(long cResult, int id);
    private native long cNewResult();
}
