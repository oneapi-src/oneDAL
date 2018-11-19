/* file: Parameter.java */
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
 * @ingroup multinomial_naive_bayes
 * @{
 */
/**
 * @brief Contains classes for computing the Naive Bayes
 */
package com.intel.daal.algorithms.multinomial_naive_bayes;

import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__PARAMETER"></a>
 * @brief Parameters for multinomial naive Bayes algorithm
 */
public class Parameter extends com.intel.daal.algorithms.classifier.Parameter {

    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Sets prior class estimates, numeric table of size [nClasses x 1]
     *  @param priorClassEstimates  Prior class estimates
     */
    public void setPriorClassEstimates(NumericTable priorClassEstimates) {
        cSetPriorClassEstimates(this.cObject, priorClassEstimates.getCObject());
    }

    /**
     *  Returnss prior class estimates, numeric table of size [nClasses x 1]
     *  @return  Prior class estimates
     */
    public NumericTable getPriorClassEstimates() {
        NumericTable nt = (NumericTable)Factory.instance().createObject(getContext(), cGetPriorClassEstimates(this.cObject));
        return nt;
    }

    /**
     *  Sets imagined occurrences of the each feature, numeric table of size [1 x nFeatures]
     *  @param alpha  Imagined occurrences of the each feature
     */
    public void setAlpha(NumericTable alpha) {
        cSetAlpha(this.cObject, alpha.getCObject());
    }

    /**
     *  Returnss imagined occurrences of the each feature, numeric table of size [1 x nFeatures]
     *  @return  Imagined occurrences of the each feature
     */
    public NumericTable getAlpha() {
        NumericTable nt = (NumericTable)Factory.instance().createObject(getContext(), cGetAlpha(this.cObject));
        return nt;
    }

    private native void cSetPriorClassEstimates(long parAddr, long ntAddr);

    private native long cGetPriorClassEstimates(long parAddr);

    private native void cSetAlpha(long parAddr, long ntAddr);

    private native long cGetAlpha(long parAddr);
}
/** @} */
