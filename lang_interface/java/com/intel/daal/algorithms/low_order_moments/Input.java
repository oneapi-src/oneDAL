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

package com.intel.daal.algorithms.low_order_moments;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__INPUT"></a>
 * @brief %Input for the low order %moments algorithm
 */
public class Input extends com.intel.daal.algorithms.Input {
    public long cAlgorithm;
    public Precision prec;
    public Method                               method;  /*!< Computation method for the algorithm */
    public ComputeMode cmode;

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public Input(DaalContext context, long cInput) {
        super(context, cInput);
    }

    public Input(DaalContext context, long cInput, long cAlgorithm, Precision prec, Method method, ComputeMode cmode,
                 ComputeStep step) {
        super(context);

        this.cObject = cInput;
        this.cAlgorithm = cAlgorithm;
        this.prec = prec;
        this.method = method;
        this.cmode = cmode;
    }

    public Input(DaalContext context, long cInput, long cAlgorithm, Precision prec, Method method, ComputeMode cmode) {
        super(context);

        this.cObject = cInput;
        this.cAlgorithm = cAlgorithm;
        this.prec = prec;
        this.method = method;
        this.cmode = cmode;
    }

    /**
     * Sets the input for the low order moments algorithm
     * @param id    Identifier of the %input object
     * @param val   The input object
     */
    public void set(InputId id, NumericTable val) {
        if (id == InputId.data) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns the input object for the low order %moments algorithm
     * @param id identifier of the input object
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

    public void setCInput(long cInput) {
        this.cObject = cInput;

        if(cmode == ComputeMode.batch) {
            cSetCInputObjectBatch(this.cObject, this.cAlgorithm, prec.getValue(), method.getValue());
        }
    }

    private native void cSetInput(long cInput, int id, long ntAddr);
    private native long cGetInputTable(long cInput, int id);
    private native void cSetCInputObjectBatch(long inputAddr, long algAddr, int prec, int method);
}
