/* file: InitPartialResult.java */
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

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.implicit_als.PartialModel;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITPARTIALRESULT"></a>
 * @brief Provides methods to access partial results of computing the initial model for the
 * implicit ALS training algorithm
 */
public final class InitPartialResult extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public InitPartialResult(DaalContext context) {
        super(context);
        this.cObject = cNewPartialResult();
    }

    public InitPartialResult(DaalContext context, long cAlgorithm, Precision precision, InitMethod method,
            ComputeMode cmode) {
        super(context);
        cObject = cGetPartialResult(cAlgorithm, precision.getValue(), method.getValue(), cmode.getValue());
    }

    /**
     * Returns a partial result of computing the initial model for the implicit ALS training algorithm
     * @param  id   Identifier of the partial result
     * @return      Partial result that corresponds to the given identifier
     */
    public PartialModel get(InitPartialResultId id) {
        if (id != InitPartialResultId.partialModel) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new PartialModel(getContext(), cGetPartialResultModel(cObject, id.getValue()));
    }

    /**
     * Sets a partial result of computing the initial model for the implicit ALS training algorithm
     * @param id    Identifier of the partial result
     * @param value Partial result that corresponds to the given identifier
     */
    public void set(InitPartialResultId id, PartialModel value) {
        int idValue = id.getValue();
        if (id != InitPartialResultId.partialModel) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetPartialResultModel(cObject, idValue, value.getCObject());
    }

    private native long cNewPartialResult();

    private native long cGetPartialResult(long cAlgorithm, int precision, int method, int mode);

    private native long cGetPartialResultModel(long cResult, int id);

    private native void cSetPartialResultModel(long cResult, int id, long cModel);
}
