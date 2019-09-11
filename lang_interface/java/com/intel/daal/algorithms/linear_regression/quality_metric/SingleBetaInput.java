/* file: SingleBetaInput.java */
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
 * @ingroup linear_regression_quality_metric_single_beta
 * @{
 */
package com.intel.daal.algorithms.linear_regression.quality_metric;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.linear_regression.Model;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLEBETAINPUT"></a>
 * @brief  Class for the input objects of the algorithm
 */
public class SingleBetaInput extends com.intel.daal.algorithms.quality_metric.QualityMetricInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public SingleBetaInput(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Sets the input object for linear regression quality metric
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(SingleBetaDataInputId id, NumericTable val) {
        if (id != SingleBetaDataInputId.expectedResponses
                && id != SingleBetaDataInputId.predictedResponses) {
            throw new IllegalArgumentException("id unsupported");
        }

        cSetInputTable(this.cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns the input object for linear regression quality metric
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(SingleBetaDataInputId id) {
        if (id == SingleBetaDataInputId.expectedResponses
                || id == SingleBetaDataInputId.predictedResponses) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetInputTable(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets an input object for linear regression quality metric
     * @param id      Identifier of the input object
     * @param val     Linear regression model
     */
    public void set(SingleBetaModelInputId id, Model val) {
        if (id != SingleBetaModelInputId.model) {
            throw new IllegalArgumentException("Incorrect PredictionInputId");
        }
        cSetInputModel(this.cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an input object for linear regression quality metric
     * @param id    Identifier of the input object
     * @return      Linear regression model
     */
    public Model get(SingleBetaModelInputId id) {
        if (id != SingleBetaModelInputId.model) {
            throw new IllegalArgumentException("id unsupported"); // error processing
        }
        return new Model(getContext(), cGetInputModel(cObject, id.getValue()));
    }

    private native void cSetInputTable(long inputAddr, int id, long ntAddr);
    private native long cGetInputTable(long cInput, int id);
    private native void cSetInputModel(long inputAddr, int id, long ntAddr);
    private native long cGetInputModel(long cObject, int id);
}
/** @} */
