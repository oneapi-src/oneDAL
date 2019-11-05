/* file: DistributedStep1LocalInput.java */
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
 * @ingroup gbt_distributed
 * @{
 */
package com.intel.daal.algorithms.gbt.regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDSTEP1LOCALINPUT"></a>
 * @brief %Input objects for the model-based training algorithm in the first step of the distributed processing mode
 */

public final class DistributedStep1LocalInput extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public DistributedStep1LocalInput(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Sets an input object for the model-based training in the first step of the distributed processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(Step1LocalNumericTableInputId id, NumericTable val) {
        if (id == Step1LocalNumericTableInputId.step1BinnedData || id == Step1LocalNumericTableInputId.step1DependentVariable ||
            id == Step1LocalNumericTableInputId.step1InputResponse || id == Step1LocalNumericTableInputId.step1InputTreeStructure ||
            id == Step1LocalNumericTableInputId.step1InputTreeOrder) {
            cSetNumericTable(this.cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns an input object for the model-based training in the first step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public NumericTable get(Step1LocalNumericTableInputId id) {
        if (id == Step1LocalNumericTableInputId.step1BinnedData || id == Step1LocalNumericTableInputId.step1DependentVariable ||
            id == Step1LocalNumericTableInputId.step1InputResponse || id == Step1LocalNumericTableInputId.step1InputTreeStructure ||
            id == Step1LocalNumericTableInputId.step1InputTreeOrder) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetNumericTable(getCObject(), id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetNumericTable(long cObject, int id, long ntAddr);
    private native long cGetNumericTable(long cObject, int id);
}
/** @} */
