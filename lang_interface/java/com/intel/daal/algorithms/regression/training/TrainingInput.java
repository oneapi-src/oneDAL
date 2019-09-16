/* file: TrainingInput.java */
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
 * @ingroup training
 * @{
 */
package com.intel.daal.algorithms.regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__REGRESSION__TRAINING__TRAININGINPUT"></a>
 * @brief  %Input objects for the regression training algorithm
 */
public class TrainingInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the input of the regression training algorithm
     * @param context   Context to manage the input of the regression training algorithm
     */
    public TrainingInput(DaalContext context) {
        super(context);
    }

    public TrainingInput(DaalContext context, long cInput) {
        super(context, cInput);
    }

    public TrainingInput(DaalContext context, long cAlgorithm, ComputeMode cmode) {
        super(context);
        this.cObject = cInit(cAlgorithm, cmode.getValue());
    }

    /**
     * Sets the input object for the regression training algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(InputId id, NumericTable val) {
        InputId.throwIfInvalid(id);
        cSetInput(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns the input object of the regression training algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(InputId id) {
        InputId.throwIfInvalid(id);
        return (NumericTable)Factory.instance().createObject(getContext(), cGetInput(this.cObject, id.getValue()));
    }

    private native long cInit(long algAddr, int cmode);

    private native void cSetInput(long inputAddr, int id, long ntAddr);

    private native long cGetInput(long inputAddr, int id);
}
/** @} */
