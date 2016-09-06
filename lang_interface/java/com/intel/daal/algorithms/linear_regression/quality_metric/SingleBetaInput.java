/* file: SingleBetaInput.java */
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
import com.intel.daal.algorithms.linear_regression.Model;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLEBETAINPUT"></a>
 * @brief  Class for the input objects of the algorithm
 */
public class SingleBetaInput extends com.intel.daal.algorithms.quality_metric.QualityMetricInput {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
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
            return new HomogenNumericTable(getContext(), cGetInputTable(cObject, id.getValue()));
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
