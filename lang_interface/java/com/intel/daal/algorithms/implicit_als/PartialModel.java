/* file: PartialModel.java */
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

package com.intel.daal.algorithms.implicit_als;

import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PARTIALMODEL"></a>
 *
 */
public class PartialModel extends com.intel.daal.algorithms.Model {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public PartialModel(DaalContext context, long cModel) {
        super(context, cModel);
    }

    /**
     * Constructs a partial implicit ALS model from the indices and factors stored in the numeric tables
     * @param context   Context to manage the partial model
     * @param factors   Numeric table containing factors stored in row-major order
     * @param indices   Numeric table containing the indices of factors
     */
    public PartialModel(DaalContext context, NumericTable factors, NumericTable indices) {
        super(context);
        this.cObject = cNewPartialModel(factors.getCObject(), indices.getCObject());
    }

    /**
     * Returns the numeric table containing factors stored in row-major order
     * @return Numeric table containing factors stored in row-major order
     */
    public NumericTable getFactors() {
        return new HomogenNumericTable(getContext(), cGetFactors(this.getCObject()));
    }

    /**
     * Returns the numeric table containing the indices of factors
     * @return Numeric table containing the indices of factors
     */
    public NumericTable getIndices() {
        return new HomogenNumericTable(getContext(), cGetIndices(this.getCObject()));
    }

    protected native long cGetFactors(long partialModelAddr);
    protected native long cGetIndices(long partialModelAddr);
    protected native long cNewPartialModel(long factorsAddr, long indicesAddr);
}
