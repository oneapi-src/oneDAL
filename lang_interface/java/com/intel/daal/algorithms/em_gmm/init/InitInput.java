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

/**
 * @brief Contains classes for initializing the EM for GMM algorithm
 */

package com.intel.daal.algorithms.em_gmm.init;

import java.io.Serializable;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__INIT__INITINPUT"></a>
 * @brief  %Input objects for the default initialization of the EM for GMM algorithm
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
     * Sets the input object for the default initialization of the EM for GMM algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public int set(InitInputId id, Serializable val) {
        if (id == InitInputId.data) {
            return cSetInput(cObject, id.getValue(), ((NumericTable) val).getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns the input object for the default initialization of the EM for GMM algorithm
     * @param id Identifier of the input object
     * @return   %Input object that corresponds to the given identifier
     */
    public NumericTable get(InitInputId id) {
        if (id == InitInputId.data) {
            return new HomogenNumericTable(getContext(), cGetInputTable(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    protected long cObject;

    private native long cInit(long algAddr, int prec, int method, int cmode);

    private native int cSetInput(long cInput, int id, long ntAddr);

    private native long cGetInputTable(long cInput, int id);

}
