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

package com.intel.daal.algorithms.implicit_als.training;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDSTEP2MASTERINPUT"></a>
 * @brief %Input objects for the implicit ALS training algorithm in the second step of the distributed processing mode
 */
public final class DistributedStep2MasterInput extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public DistributedStep2MasterInput(DaalContext context, long cAlgorithm, Precision prec, TrainingMethod method) {
        super(context);
        this.cObject = cGetInput(cAlgorithm, prec.getValue(), method.getValue());
    }

    /**
     * Adds an input object for the implicit ALS training algorithm in the second step of the distributed processing mode
     * @param id            Identifier of the input object
     * @param partialResult Value of the input object
     */
    public void add(MasterInputId id, DistributedPartialResultStep1 partialResult) {
        if (id != MasterInputId.inputOfStep2FromStep1) {
            throw new IllegalArgumentException("Incorrect id");
        }
        cAddDataCollection(this.cObject, id.getValue(), partialResult.getCObject());
    }

    /**
     * Returns an input object for the implicit ALS training algorithm in the second step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        Input object that corresponds to the given identifier
     */
    public DataCollection get(MasterInputId id) {
        if (id != MasterInputId.inputOfStep2FromStep1) {
            throw new IllegalArgumentException("Incorrect id");
        }
        return new DataCollection(getContext(), cGetDataCollection(this.cObject, id.getValue()));
    }

    private native long cGetInput(long cAlgorithm, int prec, int method);

    private native void cAddDataCollection(long cInput, int id, long partialResultAddr);
    private native long cGetDataCollection(long cInput, int id);
}
