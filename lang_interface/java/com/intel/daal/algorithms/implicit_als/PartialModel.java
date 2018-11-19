/* file: PartialModel.java */
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
 * @ingroup implicit_als
 * @{
 */
package com.intel.daal.algorithms.implicit_als;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PARTIALMODEL"></a>
 *
 */
public class PartialModel extends com.intel.daal.algorithms.Model {
    /** @private */
    static {
        LibUtils.loadLibrary();
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
        return (NumericTable)Factory.instance().createObject(getContext(), cGetFactors(this.getCObject()));
    }

    /**
     * Returns the numeric table containing the indices of factors
     * @return Numeric table containing the indices of factors
     */
    public NumericTable getIndices() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetIndices(this.getCObject()));
    }

    protected native long cGetFactors(long partialModelAddr);
    protected native long cGetIndices(long partialModelAddr);
    protected native long cNewPartialModel(long factorsAddr, long indicesAddr);
}
/** @} */
