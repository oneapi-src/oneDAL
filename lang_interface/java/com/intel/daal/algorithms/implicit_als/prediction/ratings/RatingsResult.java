/* file: RatingsResult.java */
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

package com.intel.daal.algorithms.implicit_als.prediction.ratings;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__RATINGSRESULT"></a>
 * @brief Provides methods to access the results obtained with the compute() method
 *        of the implicit ALS ratings prediction algorithm in the batch processing mode
 */
public final class RatingsResult extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Creates the result of the implicit ALS ratings prediction algorithm and attaches it to the context
     * @param context Context for managing the memory in the native part of the result object
     */
    public RatingsResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    /** @private */
    public RatingsResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /** @private */
    public RatingsResult(DaalContext context, long cAlgorithm, Precision prec, RatingsMethod method, ComputeMode cmode,
            ComputeStep step) {
        super(context);
        this.cObject = cGetResult(cAlgorithm, prec.getValue(), method.getValue(), cmode.getValue(), step.getValue());
    }

    /** @private */
    public RatingsResult(DaalContext context, long cAlgorithm, Precision prec, RatingsMethod method) {
        super(context);
        this.cObject = cGetResult(cAlgorithm, prec.getValue(), method.getValue(), ComputeMode.batch.getValue(),
                ComputeStep.step1Local.getValue());
    }

    /**
     * Returns the result of the implicit ALS ratings prediction algorithm
     * @param id    Identifier of the result, @ref RatingsResultId
     * @return      Result that corresponds to the given identifier
     */
    public NumericTable get(RatingsResultId id) {
        if (id != RatingsResultId.prediction) {
            throw new IllegalArgumentException("RatingsResultId unsupported");
        }

        return new HomogenNumericTable(getContext(), cGetNumericTable(getCObject(), id.getValue()));
    }

    /**
     * Sets the result of the implicit ALS ratings prediction algorithm
     * @param id    Identifier of the result, @ref RatingsResultId
     * @param value Prediction result
     */
    public void set(RatingsResultId id, NumericTable value) {
        if (id != RatingsResultId.prediction) {
            throw new IllegalArgumentException("RatingsResultId unsupported");
        }
        cSetNumericTable(getCObject(), id.getValue(), value.getCObject());
    }

    private native long cNewResult();

    private native long cGetResult(long algAddress, int prec, int method, int computeMode, int computeStep);

    private native long cGetNumericTable(long resAddr, int id);
    private native void cSetNumericTable(long resAddr, int id, long cNumericTable);
}
