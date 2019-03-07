/* file: TrainingDistributedStep1Local.java */
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
 * @defgroup multinomial_naive_bayes_training_distributed Distributed
 * @ingroup multinomial_naive_bayes_training
 * @{
 */
package com.intel.daal.algorithms.multinomial_naive_bayes.training;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__TRAININGDISTRIBUTEDSTEP1LOCAL"></a>
 * @brief Algorithm class for training naive Bayes model on the first step in the distributed processing mode
 * <!-- \n<a href="DAAL-REF-MULTINOMNAIVEBAYES-ALGORITHM">Multinomial naive Bayes algorithm description and usage models</a> -->
 *
 * @par References
 *      - com.intel.daal.algorithms.classifier.training.InputId class
 *      - com.intel.daal.algorithms.classifier.training.TrainingResultId class
 *      - com.intel.daal.algorithms.classifier.training.TrainingInput class
 */
public class TrainingDistributedStep1Local extends TrainingOnline {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs multinomial naive Bayes training algorithm by copying input objects and parameters
     * of another multinomial naive Bayes training algorithm
     * @param context   Context to manage the multinomial naive Bayes training
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public TrainingDistributedStep1Local(DaalContext context, TrainingDistributedStep1Local other) {
        super(context, other);
    }

    /**
     * Constructs multinomial naive Bayes training algorithm
     * @param context   Context to manage the multinomial naive Bayes training
     * @param cls       Data type to use in intermediate computations of the multinomial naive Bayes training on the first step in the distributed
     *                  processing mode,
     *                  Double.class or Float.class
     * @param method    Multinomial naive Bayes training method, @ref TrainingMethod
     * @param nClasses  Number of classes
     */
    public TrainingDistributedStep1Local(DaalContext context, Class<? extends Number> cls, TrainingMethod method,
            long nClasses) {
        super(context, cls, method, nClasses);
    }

    /**
     * Returns the newly allocated multinomial naive Bayes training algorithm
     * with a copy of input objects and parameters of this multinomial naive Bayes training algorithm
     * @param context   Context to manage the multinomial naive Bayes training
     *
     * @return The newly allocated algorithm
     */
    @Override
    public TrainingDistributedStep1Local clone(DaalContext context) {
        return new TrainingDistributedStep1Local(context, this);
    }
}
/** @} */
