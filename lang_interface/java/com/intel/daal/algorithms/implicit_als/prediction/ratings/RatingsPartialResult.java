/* file: RatingsPartialResult.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @ingroup implicit_als_prediction
 * @{
 */
package com.intel.daal.algorithms.implicit_als.prediction.ratings;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__RATINGSPARTIALRESULT"></a>
 * @brief Provides methods to access partial results obtained with the compute() method
 *        of the implicit ALS algorithm in the rating prediction stage
 */
public final class RatingsPartialResult extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs a partial result of the implicit ALS ratings prediction algorithm and attaches it to the context
     * @param context Context to manage the memory in the native part of the partial result object
     */
    public RatingsPartialResult(DaalContext context) {
        super(context);
        this.cObject = cNewPartialResult();
    }

    /** @private */
    public RatingsPartialResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns a partial result of the implicit ALS ratings prediction algorithm
     * @param id    Identifier of the partial result
     * @return      Value that corresponds to the given identifier
     */
    public RatingsResult get(RatingsPartialResultId id) {
        if (id != RatingsPartialResultId.finalResult) {
            throw new IllegalArgumentException("RatingsPartialResultId unsupported");
        }

        return new RatingsResult(getContext(), cGetResult(getCObject(), id.getValue()));
    }

    /**
     * Sets a partial result of the implicit ALS ratings prediction algorithm
     * @param id    Identifier of the partial result
     * @param value New partial result object
     */
    public void set(RatingsPartialResultId id, RatingsResult value) {
        if (id != RatingsPartialResultId.finalResult) {
            throw new IllegalArgumentException("RatingsPartialResultId unsupported");
        }
        cSetResult(getCObject(), id.getValue(), value.getCObject());
    }

    private native long cNewPartialResult();

    private native long cGetResult(long presAddr, int id);
    private native void cSetResult(long presAddr, int id, long resAddr);
}
/** @} */
