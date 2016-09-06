/* file: InitializationProcedureIface.java */
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

import com.intel.daal.services.Disposable;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__INITIALIZATIONPROCEDUREIFACE"></a>
 * @brief Abstract interface class for initialization of partial results for the correlation or variance-covariance matrix algorithm
 */
public abstract class InitializationProcedureIface implements Disposable {

    public InitializationProcedureIface() {
        this.cObject = cNewJavaInitializationProcedure();
    }

    /**
     * Initializes partial results of the correlation or variance-covariance matrix algorithm
     * @param input         %Input of the algorithm
     * @param partialResult Partial results of the algorithm
     */
    abstract public void initialize(Input input, PartialResult partialResult);

    public long getCObject() {
        return this.cObject;
    }

    /**
     * Releases memory allocated for the native initialization procedure object
     */
    @Override
    public void dispose() {
        if (this.cObject != 0) {
            cDispose(this.cObject);
            this.cObject = 0;
        }
    }

    protected long cObject;

    private native long cNewJavaInitializationProcedure();

    private native void cDispose(long cObject);
}
