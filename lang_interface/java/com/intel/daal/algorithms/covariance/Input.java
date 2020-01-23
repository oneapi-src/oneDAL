/* file: Input.java */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
 * @ingroup covariance
 * @{
 */
package com.intel.daal.algorithms.covariance;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__INPUT"></a>
 * @brief %Input objects for the correlation or variance-covariance matrix algorithm
 */
public class Input extends com.intel.daal.algorithms.Input {
    public long        cAlgorithm;
    public Precision   prec;
    public Method      method;  /*!< Computation method for the algorithm */
    public ComputeMode cmode;
    public ComputeStep step;

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Input(DaalContext context, long cInput) {
        super(context, cInput);
    }

    public Input(DaalContext context, long cAlgorithm, Precision prec, Method method, ComputeMode cmode,
                 ComputeStep step) {
        super(context);

        this.cAlgorithm = cAlgorithm;
        this.prec = prec;
        this.method = method;
        this.cmode = cmode;
        this.step = step;
        this.cObject = cInit(cAlgorithm, prec.getValue(), method.getValue(), cmode.getValue(), step.getValue());
    }

    public Input(DaalContext context, long cAlgorithm, Precision prec, Method method, ComputeMode cmode) {
        super(context);

        this.cAlgorithm = cAlgorithm;
        this.prec = prec;
        this.method = method;
        this.cmode = cmode;
        this.cObject = cInit(cAlgorithm, prec.getValue(), method.getValue(), cmode.getValue(),
                             ComputeStep.step1Local.getValue());
    }

    /**
     * Sets an input object for the correlation or variance-covariance matrix algorithm
     * @param id    Identifier of the input object
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
     * Returns an input object for the correlation or variance-covariance matrix algorithm
     * @param id Identifier of the input object
     * @return   %Input object that corresponds to the given identifier
     */
    public NumericTable get(InputId id) {
        if (id == InputId.data) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue()));
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
        else if(cmode == ComputeMode.online) {
            cSetCInputObjectOnline(this.cObject, this.cAlgorithm, prec.getValue(), method.getValue());
        }
        else if(cmode == ComputeMode.distributed) {
            if(step == ComputeStep.step1Local) {
                cSetCInputObjectDistributedStep1Local(this.cObject, this.cAlgorithm, prec.getValue(), method.getValue());
            }
        }
    }

    private native long cInit(long algAddr, int prec, int method, int cmode, int step);

    private native void cSetCInputObjectBatch(long inputAddr, long algAddr, int prec, int method);
    private native void cSetCInputObjectOnline(long inputAddr, long algAddr, int prec, int method);
    private native void cSetCInputObjectDistributedStep1Local(long inputAddr, long algAddr, int prec, int method);

    private native void cSetInput(long cInput, int id, long ntAddr);

    private native long cGetInput(long cInput, int id);
}
/** @} */
