/* file: Input.java */
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

/**
 * \brief Contains classes for ridge regression model-based prediction
 */
package com.intel.daal.algorithms.ridge_regression.prediction;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ridge_regression.Model;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.SerializableBase;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__PREDICTION__INPUT"></a>
 * @brief %Input object for making ridge regression model-based prediction
 */
public final class Input extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public Input(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets an input object for ridge regression model-based prediction
     * @param id      Identifier of the input object
     * @param val     Serializable object
     */
    public void set(PredictionInputId id, SerializableBase val) {
        if (id != PredictionInputId.data && id != PredictionInputId.model) {
            throw new IllegalArgumentException("Incorrect PredictionInputId");
        }

        long addr = 0;

        if (id == PredictionInputId.data) {
            addr = ((NumericTable) val).getCObject();
        } else if (id == PredictionInputId.model) {
            addr = ((Model) val).getCObject();
        }
        cSetInput(this.cObject, id.getValue(), addr);
    }

    /**
     * Returns an input object for ridge regression model-based prediction
     * @param id      Identifier of the input object
     * @return      Serializable object that corresponds to the given identifier
     */
    public SerializableBase get(PredictionInputId id) {
        if (id != PredictionInputId.data && id != PredictionInputId.model) {
            throw new IllegalArgumentException("id unsupported"); // error processing
        }

        if (id == PredictionInputId.data) {
            return new HomogenNumericTable(getContext(), cGetInput(cObject, id.getValue()));
        } else if (id == PredictionInputId.model) {
            return new Model(getContext(), cGetInput(cObject, id.getValue()));
        }
        return null;
    }

    private native void cSetInput(long cObject, int id, long resAddr);

    private native long cGetInput(long cObject, int id);
}
