/* file: Input.java */
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
 * @defgroup ridge_regression_prediction Prediction
 * @brief Contains a class for making ridge regression model-based prediction
 * @ingroup ridge_regression
 * @{
 */
/**
 * \brief Contains classes for ridge regression model-based prediction
 */
package com.intel.daal.algorithms.ridge_regression.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ridge_regression.Model;
import com.intel.daal.data_management.data.Factory;
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
        LibUtils.loadLibrary();
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
            return (NumericTable)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue()));
        } else if (id == PredictionInputId.model) {
            return new Model(getContext(), cGetInput(cObject, id.getValue()));
        }
        return null;
    }

    private native void cSetInput(long cObject, int id, long resAddr);

    private native long cGetInput(long cObject, int id);
}
/** @} */
