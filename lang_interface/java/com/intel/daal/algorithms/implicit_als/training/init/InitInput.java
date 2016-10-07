/* file: InitInput.java */
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

package com.intel.daal.algorithms.implicit_als.training.init;

import java.io.Serializable;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITINPUT"></a>
 * @brief Initializes input objects for the implicit ALS initialization algorithm
 */
public class InitInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public InitInput(DaalContext context, long cInput) {
        super(context);
        this.cObject = cInput;
    }

    public InitInput(DaalContext context, long cAlgorithm, Precision prec, InitMethod method, ComputeMode cmode) {
        super(context);
        this.cObject = cInit(cAlgorithm, prec.getValue(), method.getValue(), cmode.getValue());
    }

    /**
     * Sets an input object for the implicit ALS initialization algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(InitInputId id, Serializable val) {
        if (id != InitInputId.data) {
            throw new IllegalArgumentException("Incorrect InitInputId");
        }
        cSetInput(cObject, id.getValue(), ((NumericTable) val).getCObject());
    }

    /**
     * Returns an input object for the implicit ALS initialization algorithm
     * @param id    Identifier of the input object
     * @return      %Input object that corresponds to the given identifier
     */
    public NumericTable get(InitInputId id) {
        if (id != InitInputId.data) {
            throw new IllegalArgumentException("Incorrect InitInputId");
        }
        return new HomogenNumericTable(getContext(), cGetInputTable(cObject, id.getValue()));
    }

    protected long cObject;

    private native long cInit(long algAddr, int prec, int method, int cmode);

    private native void cSetInput(long cInput, int id, long ntAddr);

    private native long cGetInputTable(long cInput, int id);

}
