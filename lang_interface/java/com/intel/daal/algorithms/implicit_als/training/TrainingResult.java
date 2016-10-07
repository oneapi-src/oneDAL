/* file: TrainingResult.java */
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
import com.intel.daal.algorithms.implicit_als.Model;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__TRAININGRESULT"></a>
 * @brief Provides methods to access the results of the implicit ALS training algorithm
 */
public final class TrainingResult extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public TrainingResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public TrainingResult(DaalContext context, long cAlgorithm, Precision prec, TrainingMethod method) {
        super(context);
        this.cObject = cGetResult(cAlgorithm, prec.getValue(), method.getValue());
    }

    /**
     * Returns the result of the implicit ALS training algorithm
     * @param  id   Identifier of the result
     * @return      Result that corresponds to the given identifier
     */
    public Model get(TrainingResultId id) {
        if (id != TrainingResultId.model) {
            throw new IllegalArgumentException("TrainingResultId unsupported");
        }
        return new Model(getContext(), cGetResultModel(getCObject(), id.getValue()));
    }

    /**
     * Sets the result of the implicit ALS training algorithm
     * @param id    Identifier of the result
     * @param value Result that corresponds to the given identifier
     */
    public void set(TrainingResultId id, Model value) {
        if (id != TrainingResultId.model) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetResultModel(getCObject(), id.getValue(), value.getCObject());
    }

    private native long cNewResult();

    private native long cGetResult(long algAddr, int prec, int method);

    private native long cGetResultModel(long resAddr, int id);
    private native void cSetResultModel(long resAddr, int id, long modelAddr);
}
