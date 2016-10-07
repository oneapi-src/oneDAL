/* file: RatingsDistributedInput.java */
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
        System.loadLibrary("JavaAPI");
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
