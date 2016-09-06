/* file: DistributedStep2MasterInput.java */
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

package com.intel.daal.algorithms.covariance;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDSTEP2MASTERINPUT"></a>
 * @brief %Input objects for the correlation or variance-covariance matrix algorithm
 * in the second step of the distributed processing mode
 */
public final class DistributedStep2MasterInput extends com.intel.daal.algorithms.Input {
    public long cAlgorithm;
    public Precision prec;
    public Method                               method;  /*!< Computation method for the algorithm */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public DistributedStep2MasterInput(DaalContext context, long cAlgorithm, Precision prec, Method method) {
        super(context);
        this.cObject = cInit(cAlgorithm, prec.getValue(), method.getValue());

        this.cAlgorithm = cAlgorithm;
        this.prec = prec;
        this.method = method;
    }

    /**
     * Adds a partial result to the end of the data collection of input objects for the correlation or
     * variance-covariance matrix algorithm in the second step of the distributed processing mode
     * @param id            Identifier of the input object
     * @param pres          Partial result obtained in the first step of the distributed processing mode
     */
    public void add(DistributedStep2MasterInputId id, PartialResult pres) {
        cAddInput(cObject, id.getValue(), pres.getCObject());
    }

    public void setCInput(long cInput) {
        this.cObject = cInput;
        cSetCInputObject(this.cObject, this.cAlgorithm, prec.getValue(), method.getValue());
    }

    private native long cInit(long algAddr, int prec, int method);
    private native void cSetCInputObject(long inputAddr, long algAddr, int prec, int method);
    private native void cAddInput(long algAddr, int id, long presAddr);
}
