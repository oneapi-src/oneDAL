/* file: InitResult.java */
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
import com.intel.daal.algorithms.implicit_als.Model;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITRESULT"></a>
 * @brief Provides methods to access the results of computing the initial model for the
 * implicit ALS training algorithm
 */
public final class InitResult extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public InitResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public InitResult(DaalContext context, long cAlgorithm, Precision precision, InitMethod method, ComputeMode cmode) {
        super(context);
        cObject = cGetResult(cAlgorithm, precision.getValue(), method.getValue(), cmode.getValue());
    }

    /**
     * Returns the result of computing the initial model for the implicit ALS training algorithm
     * @param id   Identifier of the result
     * @return         Result that corresponds to the given identifier
     */
    public Model get(InitResultId id) {
        if (id != InitResultId.model) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new Model(getContext(), cGetResultModel(cObject, id.getValue()));
    }

    /**
     * Sets the result of computing the initial model for the implicit ALS training algorithm
     * @param id    Identifier of the result
     * @param value Result that corresponds to the given identifier
     */
    public void set(InitResultId id, Model value) {
        int idValue = id.getValue();
        if (id != InitResultId.model) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetResultModel(cObject, idValue, value.getCObject());
    }

    private native long cNewResult();

    private native long cGetResult(long cAlgorithm, int precision, int method, int mode);

    private native long cGetResultModel(long cResult, int id);

    private native void cSetResultModel(long cResult, int id, long cModel);
}
