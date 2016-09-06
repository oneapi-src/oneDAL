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

package com.intel.daal.algorithms.pca;

import java.io.Serializable;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__INPUT"></a>
 * @brief %Input objects for the PCA algorithm
 */
public class Input extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public Input(DaalContext context, long cInput, ComputeMode cmode) {
        super(context, cInput);
        this.cmode = cmode;
    }

    /**
     * Sets an input object for the PCA algorithm
     * @param id    Identifier of the input object
     * @param val   Serializable object
     */
    public void set(InputId id, Serializable val) throws IllegalArgumentException {
        if (id != InputId.data && id != InputId.correlation) {
            throw new IllegalArgumentException("Incorrect InputId");
        }
        if ((id == InputId.correlation) && ((cmode == ComputeMode.distributed) || (cmode == ComputeMode.online))) {
            throw new IllegalArgumentException("Incorrect InputId");
        }
        cSetInput(cObject, id.getValue(), ((NumericTable) val).getCObject());
    }

    /**
     * Returns an input object for the PCA algorithm
     * @param id Identifier of the input object
     * @return   %Input object that corresponds to the given identifier
     */
    public NumericTable get(InputId id) {
        if (id == InputId.data) {
            return new HomogenNumericTable(getContext(), cGetInputTable(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private ComputeMode cmode;

    private native void cSetInput(long cInput, int id, long ntAddr);

    private native long cGetInputTable(long cInput, int id);
}
