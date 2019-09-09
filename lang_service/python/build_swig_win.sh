#! /bin/bash
#===============================================================================
# Copyright 2014-2019 Intel Corporation
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
#===============================================================================

# Script for building SWIG on Windows with MinGW #
##################################################

# You must have the following MinGW packages installed:
#	 mingw-developer-toolkit
#	 mingw32-base
#	 mingw32-gcc-g++
#	 msys-base

# Make sure you have the following env variable set:
#    FTP_PROXY=http://proxy-chain.intel.com:911

# Clone the daal repo and and make sure the daal/swig/swig submodule
# contains the swig files.  You may need to run:
#    `git submodule init`
#    `git submodule update`

# Make sure the swig submodule is on the correct branch

# Run this script in a MinGW shell (C:\MinGW\msys\1.0\msys.bat)

# If it complains about missing commands, add C:\MinGW\bin to your PATH.

cd swig

curl -o pcre-8.38.tar.gz ftp://ftp.csx.cam.ac.uk/pub/software/programming/pcre/pcre-8.38.tar.gz

Tools/pcre-build.sh

./autogen.sh
./configure
make
