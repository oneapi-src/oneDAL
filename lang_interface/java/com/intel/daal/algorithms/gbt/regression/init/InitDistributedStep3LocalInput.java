/* file: InitDistributedStep3LocalInput.java */
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
 * @ingroup dbscan_distributed
 * @{
 */
package com.intel.daal.algorithms.gbt.regression.init;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__InitDistributedStep3LocalInput"></a>
 * @brief %Input objects for the DBSCAN algorithm in the first step of the distributed processing mode
 */

public final class InitDistributedStep3LocalInput extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public InitDistributedStep3LocalInput(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Sets an input object for the DBSCAN algorithm in the first step of the distributed processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(InitStep3LocalNumericTableInputId id, NumericTable val) {
        if (id == InitStep3LocalNumericTableInputId.step3MergedBinBorders || id == InitStep3LocalNumericTableInputId.step3BinQuantities ||
            id == InitStep3LocalNumericTableInputId.step3LocalData || id == InitStep3LocalNumericTableInputId.step3InitialResponse) {
            cSetNumericTable(this.cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns an input object for the DBSCAN algorithm in the first step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public NumericTable get(InitStep3LocalNumericTableInputId id) {
        if (id == InitStep3LocalNumericTableInputId.step3MergedBinBorders || id == InitStep3LocalNumericTableInputId.step3BinQuantities ||
            id == InitStep3LocalNumericTableInputId.step3LocalData || id == InitStep3LocalNumericTableInputId.step3InitialResponse) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetNumericTable(getCObject(), id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetNumericTable(long cObject, int id, long ntAddr);
    private native long cGetNumericTable(long cObject, int id);
}
/** @} */
