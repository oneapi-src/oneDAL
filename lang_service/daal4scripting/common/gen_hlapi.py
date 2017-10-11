#*******************************************************************************
# Copyright 2014-2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#******************************************************************************/

description = """
A tool to create SWIG interface files for HLAPI of DAAL.
See pare.py for details about C++ parsing.
See swig_interface.py for details about extracting necessary data and creating internal data strucutres.
See wrappers.py for necessary configuration that can not be auto-extracted.
See wrapper_gen.py for code generation (SWIG interface template file).
"""

from swig_interface import swig_interface
from os.path import join as jp

if __name__ == "__main__":
    import argparse

    argParser = argparse.ArgumentParser(prog="gen_hlapi.py",
                                        description=description,
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argParser.add_argument('--hlapi',    default=None, choices=['R', 'Python'], help="creates high level API for given language")
    argParser.add_argument('--daalroot', default=None,                          help="DAAL root directory (reads include dir in there)")

    args = argParser.parse_args()

    iface = swig_interface(jp(args.daalroot, 'include'))
    iface.read()
    iface.hlapi(args.hlapi, ['kmeans',
                             'svd',
                             'multinomial_naive_bayes',
                             'pca',
                             'linear_regression',
                             'multivariate_outlier_detection',
                             'univariate_outlier_detection',
                             'svm',
                             'kernel_function',
                             'multi_class_classifier']
                # 'ridge_regression',
            )
