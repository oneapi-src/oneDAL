/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <algorithm>
#include <vector>

#include <sys/time.h>

static void sync(){}

/// Internal timer object
typedef struct timer {
      struct timeval val;

    static timer start();

    static double stop();
    static double stop(timer start);

} timer;

//double timeit(void(*fn)());
// get current time
static inline timer time_now(void) {
    timer time;
    gettimeofday(&time.val, NULL);
    return time;
}

// absolute difference between two times (in seconds)
static inline double time_seconds(timer start, timer end) {
    struct timeval elapsed;
    timersub(&start.val, &end.val, &elapsed);
    long sec  = elapsed.tv_sec;
    long usec = elapsed.tv_usec;
    double t  = sec + usec * 1e-6;
    return t >= 0 ? t : -t;
}

thread_local timer _timer_;

timer timer::start() { return _timer_ = time_now(); }
double timer::stop(timer start) { return time_seconds(start, time_now()); }
double timer::stop() { return time_seconds(_timer_, time_now()); }

template<typename T>
double timeit(T& obj) {
    // parameters
    static const int trials      = 10;  // trial runs
    static const int s_trials    = 5;   // trial runs
    static const double min_time = 1;   // seconds

    std::vector<double> sample_times(s_trials);

    // estimate time for a few samples
    for (int i = 0; i < s_trials; ++i) {
        sync();
        timer start = timer::start();
        obj.compute();
        sync();
        sample_times[i] = timer::stop(start);
    }

    // Sort sample times and select the median time
    std::sort(sample_times.begin(), sample_times.end());

    double median_time = sample_times[s_trials / 2];

    // Run a bunch of batches of fn
    // Each batch runs trial runs before sync
    // If trials * median_time < min time,
    //   then run (min time / (trials * median_time)) batches
    // else
    //   run 1 batch
    int batches     = (int)ceilf(min_time / (trials * median_time));
    double run_time = 0;

    for (int b = 0; b < batches; b++) {
        timer start = timer::start();
        for (int i = 0; i < trials; ++i) obj.compute();
        sync();
        run_time += timer::stop(start) / trials;
    }
    return run_time / batches;
}

template<typename T>
double timeitM(T& obj, double& firstIt, const size_t nTimes)
{
    const double left = 0.25;
    const double right = 0.75;

    std::vector<double> times(nTimes);

    for (size_t i = 0; i < nTimes; i++)
    {
        sync();
        timer start = timer::start();
        obj();
        sync();
        times[i] = timer::stop(start);
    }

    firstIt = times[0];

    const size_t n = nTimes;
    std::vector<double> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());

    const double Q1 = sorted_times[size_t(n*left)];
    const double Q3 = sorted_times[size_t(n*right)];

    const double IQ = Q3 - Q1;

    const double lower = Q1 - 1.5 * IQ;
    const double upper = Q3 + 1.5 * IQ;

    double sum = 0.0;
    double count = 0.0;

    for (int i = 0; i < n; i++) {
        const double timei = sorted_times[i];
        if ((lower <= timei) && (timei <= upper)) {
            count++;
            sum += timei;
        }
    }

    return sum/count;
}

static double BoxFilter(const std::vector<double>& times)
{
    const double left = 0.25;
    const double right = 0.75;

    const size_t n = times.size();
    std::vector<double> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());

    const double Q1 = sorted_times[size_t(n*left)];
    const double Q3 = sorted_times[size_t(n*right)];

    const double IQ = Q3 - Q1;

    const double lower = Q1 - 1.5 * IQ;
    const double upper = Q3 + 1.5 * IQ;

    double sum = 0.0;
    double count = 0.0;

    for (int i = 0; i < n; i++)
    {
        const double timei = sorted_times[i];
        if ((lower <= timei) && (timei <= upper))
        {
            count++;
            sum += timei;
        }
    }
    return sum/count;
}

static double FirstIteration(const std::vector<double>& times)
{
    return times[0];
}
