/* file: Input.java */
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
 * @defgroup ridge_regression_training Training
 * @brief Contains a class for ridge regression model-based training
 * @ingroup ridge_regression
 * @{
 */
package com.intel.daal.algorithms.ridge_regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__TRAINING__INPUT"></a>
 * @brief %Input object for ridge regression model-based training in the batch and
 * online processing modes and in the first step of the distributed processing mode
 */
public class Input extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Input(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets an input object for ridge regression model-based training
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(TrainingInputId id, NumericTable val) {
        if ((id != TrainingInputId.data) && (id != TrainingInputId.dependentVariable)) {
            throw new IllegalArgumentException("Incorrect TrainingInputId");
        }

        cSetInput(this.cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an input object for ridge regression model-based training
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public NumericTable get(TrainingInputId id) {
        if ((id != TrainingInputId.data) && (id != TrainingInputId.dependentVariable)) {
            throw new IllegalArgumentException("id unsupported"); // error processing
        }

        return (NumericTable)Factory.instance().createObject(getContext(), cGetInput(this.cObject, id.getValue()));
    }

    private native void cSetInput(long cObject, int id, long resAddr);

    private native long cGetInput(long cObject, int id);
}
/** @} */
