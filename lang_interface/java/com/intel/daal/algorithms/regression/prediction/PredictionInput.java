/* file: PredictionInput.java */
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
 * @ingroup prediction
 * @{
 */
package com.intel.daal.algorithms.regression.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.regression.Model;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__REGRESSION__PREDICTION__PREDICTIONINPUT"></a>
 * @brief  %Input objects for the regression algorithm
 */
public class PredictionInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the input of the regression model-based prediction algorithm
     * @param context   Context to manage the input of the regression model-based prediction algorithm
     */
    public PredictionInput(DaalContext context) {
        super(context);
    }

    public PredictionInput(DaalContext context, long cAlgorithm) {
        super(context);
        this.cObject = cInit(cAlgorithm);
    }

    public PredictionInput(DaalContext context, long cAlgorithm, ComputeMode cmode) {
        super(context);
        this.cObject = cInit(cAlgorithm);
    }

    /**
     * Sets the NumericTable input object for the regression model-based prediction algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(NumericTableInputId id, NumericTable val) {
        NumericTableInputId.throwIfInvalid(id);
        cSetInputTable(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns the NumericTable input object for the regression model-based prediction algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(NumericTableInputId id) {
        NumericTableInputId.throwIfInvalid(id);
        return (NumericTable)Factory.instance().createObject(
                getContext(), cGetInputTable(cObject, id.getValue()));
    }

    /**
     * Sets the Model input object for the regression model-based prediction algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(ModelInputId id, Model val) {
        ModelInputId.throwIfInvalid(id);
        cSetInputModel(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns the Model input object for the regression model-based prediction algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Model get(ModelInputId id) {
        ModelInputId.throwIfInvalid(id);
        return new Model(getContext(), cGetInputModel(cObject, id.getValue()));
    }

    private native void cSetInputTable(long inputAddr, int id, long ntAddr);
    private native long cGetInputTable(long cInput, int id);
    private native void cSetInputModel(long inputAddr, int id, long modelAddr);
    protected native long cGetInputModel(long inputAddr, int id);
    private native long cInit(long inputAddr);
}
/** @} */
