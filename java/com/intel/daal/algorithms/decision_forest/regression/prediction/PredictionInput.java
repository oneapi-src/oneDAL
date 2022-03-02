/* file: PredictionInput.java */
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
 * @ingroup decision_forest_regression_prediction
 * @{
 */
package com.intel.daal.algorithms.decision_forest.regression.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.decision_forest.regression.Model;
import com.intel.daal.algorithms.decision_forest.regression.prediction.NumericTableInputId;
import com.intel.daal.algorithms.decision_forest.regression.prediction.ModelInputId;
import com.intel.daal.algorithms.Precision;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__REGRESSION__PREDICTION__PREDICTIONINPUT"></a>
 * @brief  %Input objects for the decision forest regression prediction algorithm
 */
public class PredictionInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public PredictionInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    public PredictionInput(DaalContext context, long cAlgorithm, Precision prec, PredictionMethod method) {
        super(context);
        this.cObject = cInit(cAlgorithm, prec.getValue(), method.getValue());
    }

    /**
     * Sets the NumericTable input object for the decision forest regression model-based prediction algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(NumericTableInputId id, NumericTable val) {
        if (id != NumericTableInputId.data) {
            throw new IllegalArgumentException("id unsupported");
        }

        cSetInputTable(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns the NumericTable input object for the decision forest regression model-based prediction algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(NumericTableInputId id) {
        if (id == NumericTableInputId.data) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetInputTable(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the Model input object for the decision forest regression model-based prediction algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(ModelInputId id, Model val) {
        if (id != ModelInputId.model) {
            throw new IllegalArgumentException("id unsupported");
        }

        cSetInputModel(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns the Model input object for the decision forest regression model-based prediction algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Model get(ModelInputId id) {
        if (id == ModelInputId.model) {
            return new Model(getContext(), cGetInputModel(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInputTable(long inputAddr, int id, long ntAddr);

    private native long cGetInputTable(long cInput, int id);

    private native void cSetInputModel(long inputAddr, int id, long modelAddr);

    private native long cGetInputModel(long inputAddr, int id);

    private native long cInit(long algAddr, int prec, int method);

}
/** @} */
