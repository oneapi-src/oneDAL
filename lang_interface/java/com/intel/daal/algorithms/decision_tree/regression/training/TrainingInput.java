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
package com.intel.daal.algorithms.decision_tree.regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.Precision;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__REGRESSION__TRAINING__TRAININGINPUT"></a>
 * @brief  %Input objects for the decision_tree regression algorithm model training
 */
public class TrainingInput extends com.intel.daal.algorithms.regression.training.TrainingInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public TrainingInput(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Sets the input object for the decision_tree regression model training algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(TrainingInputId id, NumericTable val) {
        TrainingInputId.throwIfInvalid(id);
        cSetInput(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns the input object of the decision_tree regression model training algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(TrainingInputId id) {
        TrainingInputId.throwIfInvalid(id);
        return (NumericTable)Factory.instance().createObject(getContext(), cGetInput(this.cObject, id.getValue()));
    }

    /**
     * Returns the number of columns in the input data set
     * @return Number of columns in the input data set
     */
    public long getNumberOfFeatures() {
        return cGetNumberOfFeatures(cObject);
    }

    /**
     * Returns the number of dependent variables
     * @return Number of dependent variables
     */
    public long getNumberOfDependentVariables() {
        return cGetNumberOfDependentVariables(cObject);
    }

    private native void cSetInput(long inputAddr, int id, long ntAddr);

    private native long cGetInput(long inputAddr, int id);

    private native long cGetNumberOfFeatures(long inputAddr);

    private native long cGetNumberOfDependentVariables(long inputAddr);
}
/** @} */
