/* file: RatingsInput.java */
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
 * @ingroup implicit_als_prediction_batch
 * @{
 */
package com.intel.daal.algorithms.implicit_als.prediction.ratings;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.implicit_als.Model;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__RATINGSINPUT"></a>
 * @brief %Input objects for the rating prediction stage of the implicit ALS algorithm
 * in the batch processing mode
 */
public class RatingsInput extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public RatingsInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    public RatingsInput(DaalContext context, long cAlgorithm, Precision prec, RatingsMethod method) {
        super(context);
        this.cObject = cInit(cAlgorithm, prec.getValue(), method.getValue());
    }

    /**
     * Sets an input model object for the rating prediction stage of the implicit ALS algorithm
     * in the batch processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(RatingsModelInputId id, Model val) {
        if (id != RatingsModelInputId.model) {
            throw new IllegalArgumentException("Incorrect RatingsModelInputId");
        }
        cSetModel(this.cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an input model object for the rating prediction stage of the implicit ALS algorithm
     * in the batch processing mode
     * @param id      Identifier of the input Model object
     * @return        Input object that corresponds to the given identifier
     */
    public Model get(RatingsModelInputId id) {
        if (id != RatingsModelInputId.model) {
            throw new IllegalArgumentException("Incorrect RatingsModelInputId"); // error processing
        }
        return new Model(getContext(), cGetModel(this.cObject, id.getValue()));
    }

    private native long cInit(long algAddr, int prec, int method);

    private native void cSetModel(long cObject, int id, long modelAddr);
    private native long cGetModel(long cObject, int id);
}
/** @} */
