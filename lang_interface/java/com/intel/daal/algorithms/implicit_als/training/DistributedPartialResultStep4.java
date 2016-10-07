/* file: DistributedPartialResultStep4.java */
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
import com.intel.daal.algorithms.implicit_als.PartialModel;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP4"></a>
 * @brief Provides methods to access partial results obtained with the compute() method of the
 *        implicit ALS training algorithm in the fourth step of the distributed processing mode
 */
public final class DistributedPartialResultStep4 extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Creates a partial result of the implicit ALS training algorithm from the context
     * @param context Context for managing the memory in the native part of the partial result object
    */
    public DistributedPartialResultStep4(DaalContext context) {
        super(context);
        this.cObject = cNewDistributedPartialResultStep4();
    }

    public DistributedPartialResultStep4(DaalContext context, long cAlgorithm, Precision prec, TrainingMethod method) {
        super(context);
        this.cObject = cGetDistributedPartialResultStep4(cAlgorithm, prec.getValue(), method.getValue());
    }

    /**
     * Returns a partial result of the implicit ALS training algorithm obtained in the fourth step of the distributed processing mode
     * @param  id   Identifier of the input object, @ref DistributedPartialResultStep4Id
     * @return Partial result that corresponds to the given identifier
     */
    public PartialModel get(DistributedPartialResultStep4Id id) {
        int idValue = id.getValue();
        if (id != DistributedPartialResultStep4Id.outputOfStep4ForStep1 &&
            id != DistributedPartialResultStep4Id.outputOfStep4ForStep3 &&
            id != DistributedPartialResultStep4Id.outputOfStep4) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new PartialModel(getContext(), cGetPartialModel(getCObject(), idValue));
    }

    /**
    * Sets a partial result of the implicit ALS training algorithm obtained in the fourth step of the distributed processing mode
    * @param id     Identifier of the input object
    * @param value  Value of the input object
    */
    public void set(DistributedPartialResultStep4Id id, PartialModel value) {
        int idValue = id.getValue();
        if (id != DistributedPartialResultStep4Id.outputOfStep4ForStep1 &&
            id != DistributedPartialResultStep4Id.outputOfStep4ForStep3 &&
            id != DistributedPartialResultStep4Id.outputOfStep4) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetPartialModel(getCObject(), idValue, value.getCObject());
    }

    private native long cNewDistributedPartialResultStep4();
    private native long cGetDistributedPartialResultStep4(long cAlgorithm, int prec, int method);

    private native long cGetPartialModel(long cObject, int id);
    private native void cSetPartialModel(long cObject, int id, long cPartialModel);
}
