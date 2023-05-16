/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#pragma once

// Enables Catch2 microbenchmarking API
// https://github.com/catchorg/Catch2/blob/devel/docs/benchmarks.md
#define CATCH_CONFIG_ENABLE_BENCHMARKING

// CATCH_CONFIG_POSIX_SIGNAL enables handling of POSIX signals.
// For unknown reason user-defined handlers for signals
// (see https://en.wikipedia.org/wiki/C_signal_handling)
// catches SIGSEGV signal when USM pointer is accessed on host.
// To make USM work, we disable signal handling in Catch2.
#define CATCH_CONFIG_NO_POSIX_SIGNALS

// Disables unexpected exceptions handing in Catch2.
// It is easier to debug exception via GDB if there is handler.
#define CATCH_CONFIG_DISABLE_EXCEPTIONS

#include <catch2/catch.hpp>
