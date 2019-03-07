/* file: ConcatParameter.java */
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
 * @ingroup concat
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.concat;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__CONCATPARAMETER"></a>
 * \brief Class that specifies parameters of the concat layer
 */
public class ConcatParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {

    /**
     *  Constructs the parameters for the concat layer
     */
    public ConcatParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public ConcatParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Gets the index of dimension along which concatenation is implemented
     */
    public long getConcatDimension() {
        return cGetConcatDimension(cObject);
    }

    /**
     *  Sets the index of dimension along which concatenation is implemented
     *  @param concatDimension ConcatIndex of dimension along which concatenation is implemented
     */
    public void setConcatDimension(long concatDimension) {
       cSetConcatDimension(cObject, concatDimension);
    }

    private native long cInit();
    private native long cGetConcatDimension(long cParameter);
    private native void cSetConcatDimension(long cParameter, long concatDimension);
}
/** @} */
