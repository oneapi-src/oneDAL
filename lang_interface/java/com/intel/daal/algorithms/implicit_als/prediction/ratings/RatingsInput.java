/* file: RatingsInput.java */
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
        System.loadLibrary("JavaAPI");
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
