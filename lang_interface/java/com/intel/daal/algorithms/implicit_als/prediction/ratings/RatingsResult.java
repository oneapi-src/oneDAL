/* file: RatingsResult.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/**
 * @ingroup implicit_als_prediction_batch
 * @{
 */
package com.intel.daal.algorithms.implicit_als.prediction.ratings;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
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
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the result of the implicit ALS ratings prediction algorithm and attaches it to the context
     * @param context Context to manage the memory in the native part of the result object
     */
    public RatingsResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    /** @private */
    public RatingsResult(DaalContext context, long cObject) {
        super(context, cObject);
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

        return (NumericTable)Factory.instance().createObject(getContext(), cGetNumericTable(getCObject(), id.getValue()));
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

    private native long cGetNumericTable(long resAddr, int id);
    private native void cSetNumericTable(long resAddr, int id, long cNumericTable);
}
/** @} */
