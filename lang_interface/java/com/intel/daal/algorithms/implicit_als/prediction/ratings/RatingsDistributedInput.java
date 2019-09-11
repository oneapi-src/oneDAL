/* file: RatingsDistributedInput.java */
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
 * @ingroup implicit_als_prediction_distributed
 * @{
 */
package com.intel.daal.algorithms.implicit_als.prediction.ratings;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.implicit_als.PartialModel;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PREDICTION__RATINGS__RATINGSDISTRIBUTEDINPUT"></a>
 * @brief %Input objects for the first step of the rating prediction stage of the implicit ALS algorithm
 * in the distributed processing mode
 */
public class RatingsDistributedInput extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public RatingsDistributedInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    public RatingsDistributedInput(DaalContext context, long cAlgorithm, Precision prec, RatingsMethod method) {
        super(context);
        this.cObject = cInit(cAlgorithm, prec.getValue(), method.getValue());
    }

    /**
     * Sets an input partial model object for the rating prediction stage of the implicit ALS algorithm
     * in the distributed processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(RatingsPartialModelInputId id, PartialModel val) {
        if (id != RatingsPartialModelInputId.usersPartialModel &&
            id != RatingsPartialModelInputId.itemsPartialModel) {
            throw new IllegalArgumentException("Incorrect RatingsPartialModelInputId");
        }
        cSetPartialModel(this.cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an input partial model object for the rating prediction stage of the implicit ALS algorithm
     * in the distributed processing mode
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public PartialModel get(RatingsPartialModelInputId id) {
        if (id != RatingsPartialModelInputId.usersPartialModel &&
            id != RatingsPartialModelInputId.itemsPartialModel) {
            throw new IllegalArgumentException("Incorrect RatingsPartialModelInputId"); // error processing
        }
        return new PartialModel(getContext(), cGetPartialModel(this.cObject, id.getValue()));
    }

    private native long cInit(long algAddr, int prec, int method);

    private native void cSetPartialModel(long cObject, int id, long modelAddr);
    private native long cGetPartialModel(long cObject, int id);
}
/** @} */
