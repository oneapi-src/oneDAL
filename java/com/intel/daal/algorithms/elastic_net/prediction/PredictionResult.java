/* file: PredictionResult.java */
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
 * @ingroup elastic_net_prediction
 * @{
 */
package com.intel.daal.algorithms.elastic_net.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ELASTIC_NET__PREDICTION__PREDICTIONRESULT"></a>
 * @brief Result object for elastic net model-based prediction
 */
public final class PredictionResult extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the elastic net prediction result
     * @param context   Context to manage the  result of the elastic net prediction algorithm
     */
    public PredictionResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public PredictionResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of elastic net model-based prediction
     * @param id    Identifier of the result
     * @return      Result that corresponds to the given identifier
     */
    public NumericTable get(PredictionResultId id) {
        int idValue = id.getValue();
        if (idValue != PredictionResultId.prediction.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }

        return (NumericTable)Factory.instance().createObject(getContext(), cGetPredictionResult(cObject, idValue));
    }

    /**
     * Sets the result of elastic net model-based prediction
     * @param id    Identifier of the result
     * @param val   Result that corresponds to the given identifier
     */
    public void set(PredictionResultId id, NumericTable val) {
        int idValue = id.getValue();
        if (idValue != PredictionResultId.prediction.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetPredictionResult(cObject, idValue, val.getCObject());
    }

    private native long cNewResult();

    private native long cGetPredictionResult(long resAddr, int id);

    private native void cSetPredictionResult(long cObject, int id, long cNumericTable);

}
/** @} */
