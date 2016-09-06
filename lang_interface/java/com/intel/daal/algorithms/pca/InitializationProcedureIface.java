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

package com.intel.daal.algorithms.pca;

import com.intel.daal.services.Disposable;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__INITIALIZATIONPROCEDUREIFACE"></a>
 * @brief Abstract interface class for partial results initialization procedure
 */
public abstract class InitializationProcedureIface implements Disposable {

    public InitializationProcedureIface(Method method) {
        this.cObject = cNewJavaInitializationProcedure(method.getValue());
    }

    /**
     * Initialize partial results
     * @param input         Input parameters of the PCA algorithm
     * @param partialResult Partial results of the PCA algorithm
     */
    abstract public void initialize(Input input, com.intel.daal.algorithms.PartialResult partialResult);

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

    private long cObject;

    private native long cNewJavaInitializationProcedure(int method);

    private native void cDispose(long cObject);
}
