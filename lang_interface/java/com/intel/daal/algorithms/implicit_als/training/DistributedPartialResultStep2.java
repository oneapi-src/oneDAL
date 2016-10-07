/* file: DistributedPartialResultStep2.java */
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

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP2"></a>
 * @brief Provides methods to access partial results obtained with the compute() method of the
 *        implicit ALS training algorithm in the second step of the distributed processing mode
 */
public final class DistributedPartialResultStep2 extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Creates a partial result of the implicit ALS training algorithm from the context
     * @param context Context for managing the memory in the native part of the partial result object
    */
    public DistributedPartialResultStep2(DaalContext context) {
        super(context);
        this.cObject = cNewDistributedPartialResultStep2();
    }

    public DistributedPartialResultStep2(DaalContext context, long cAlgorithm, Precision prec, TrainingMethod method) {
        super(context);
        this.cObject = cGetDistributedPartialResultStep2(cAlgorithm, prec.getValue(), method.getValue());
    }

    /**
     * Returns a partial result of the implicit ALS training algorithm obtained in the second step of the distributed processing mode
     * @param  id   Identifier of the input object, @ref DistributedPartialResultStep2Id
     * @return Partial result that corresponds to the given identifier
     */
    public NumericTable get(DistributedPartialResultStep2Id id) {
        if (id != DistributedPartialResultStep2Id.outputOfStep2ForStep4) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new HomogenNumericTable(getContext(), cGetNumericTable(getCObject(), id.getValue()));
    }

    /**
    * Sets a partial result of the implicit ALS training algorithm obtained in the second step of the distributed processing mode
    * @param id       Identifier of the input object
    * @param value    Value of the input object
    */
    public void set(DistributedPartialResultStep2Id id, NumericTable value) {
        if (id != DistributedPartialResultStep2Id.outputOfStep2ForStep4) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetNumericTable(getCObject(), id.getValue(), value.getCObject());
    }

    private native long cNewDistributedPartialResultStep2();
    private native long cGetDistributedPartialResultStep2(long cAlgorithm, int prec, int method);

    private native long cGetNumericTable(long cObject, int id);
    private native void cSetNumericTable(long cObject, int id, long cNumericTable);
}
