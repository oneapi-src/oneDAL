.. ******************************************************************************
.. * Copyright 2014-2020 Intel Corporation
.. *
.. * Licensed under the Apache License, Version 2.0 (the "License");
.. * you may not use this file except in compliance with the License.
.. * You may obtain a copy of the License at
.. *
.. *     http://www.apache.org/licenses/LICENSE-2.0
.. *
.. * Unless required by applicable law or agreed to in writing, software
.. * distributed under the License is distributed on an "AS IS" BASIS,
.. * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. * See the License for the specific language governing permissions and
.. * limitations under the License.
.. *******************************************************************************/

.. re-use for math equations:
.. |x| replace:: :math:`x`
.. |y| replace:: :math:`y`

Density-Based Spatial Clustering of Applications with Noise
===========================================================

Density-based spatial clustering of applications with noise (DBSCAN) is a data clustering algorithm proposed in [Ester96]_. 
It is a density-based clustering non-parametric algorithm: given a set of observations in some space,
it groups together observations that are closely packed together (observations with many nearby neighbors),
marking as outliers observations that lie alone in low-density regions (whose nearest neighbors are too far away).

Details
*******

Given the set :math:`X = \{x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})\}` 
of :math:`n` :math:`p`-dimensional feature vectors (further referred as observations),
a positive floating-point number epsilon and a positive integer minObservations,
the problem is to get clustering assignments for each input observation, based on the definitions below [Ester96]_:

An observation |x| is called core observation if at least ``minObservations``
input observations (including |x|) are within distance epsilon from observation |x|;

An observation |y| is directly reachable from |x| if |y| is within distance epsilon from core observation |x|. 
Observations are only said to be directly reachable from core observations.

An observation |y| is reachable from observation |x| if there is a path :math:`x_1, \ldots, x_m` 
with :math:`x_1 = x` and :math:`x_m = y`, where each :math:`x_{i+1}` is directly reachable from :math:`x_i`.
This implies that all observations on the path must be core observations, with the possible exception of |y|.

All observations not reachable from any other observation are noise observations.

Two observations |x| and |y| is considered to be in the same cluster if there is a core observation :math:`z`, 
that |x| and |y| are reachable from :math:`z`.

Each cluster will get a unique identifier, an integer number from :math:`0` to (total number of clusters – :math:`1`).
Assignment of each observation is an identifier of the cluster to which it belongs, 
or :math:`-1` if the observation considered to be a noise observation.

Computation
***********

The following computation modes are available:

.. toctree::
   :maxdepth: 1
   
   computation-batch.rst
   computation-distributed.rst
