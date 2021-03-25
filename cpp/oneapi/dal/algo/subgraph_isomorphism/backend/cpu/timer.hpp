#ifndef TIMER_H
#define TIMER_H

typedef struct timeval TIMEHANDLE;

extern struct timezone _tz;
inline TIMEHANDLE start_time() {
    TIMEHANDLE tstart;
    gettimeofday(&tstart, &_tz);
    return tstart;
}
inline double end_time(TIMEHANDLE th) {
    TIMEHANDLE tend;
    double t1, t2;

    gettimeofday(&tend, &_tz);

    t1 = (double)th.tv_sec + (double)th.tv_usec / (1000 * 1000);
    t2 = (double)tend.tv_sec + (double)tend.tv_usec / (1000 * 1000);
    return t2 - t1;
}

#endif
