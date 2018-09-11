/* file: GroupOfBetasResult.java */
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
 * @ingroup linear_regression_quality_metric_group_of_betas
 * @{
 */
package com.intel.daal.algorithms.linear_regression.quality_metric;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__GROUPOFBETASRESULT"></a>
 * @brief  Class for the the result of linear regression quality metrics algorithm
 */
public class GroupOfBetasResult extends com.intel.daal.algorithms.quality_metric.QualityMetricResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public GroupOfBetasResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Constructs the result of the quality metric algorithm
     * @param context   Context to manage the result of the quality metric algorithm
     */
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
            return (NumericTable)Factory.instance().createObject(getContext(), cGetResultTable(cObject, id.getValue()));
        throw new IllegalArgumentException("id unsupported");
    }

    private native void cSetResultTable(long inputAddr, int id, long ntAddr);
    private native long cGetResultTable(long cResult, int id);
    private native long cNewResult();
}
/** @} */
