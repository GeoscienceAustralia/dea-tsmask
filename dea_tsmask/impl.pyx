import cython

import numpy
cimport numpy

from libc.math cimport isnan, round

ctypedef numpy.uint8_t uint8
ctypedef numpy.float32_t float32

cdef uint8 NO_OBS = 0
cdef uint8 CLEAR = 1
cdef uint8 CLOUD = 2
cdef uint8 SHADOW = 3

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void initial_guess(const float32[:, :, :] avg, uint8[:, :, :] mask, float32 brightness_threshold) nogil:
    cdef int t, y, x

    for t in range(avg.shape[0]):
        for y in range(avg.shape[1]):
            for x in range(avg.shape[2]):
                if not isnan(avg[t, y, x]):
                    if avg[t, y, x] > brightness_threshold:
                        mask[t, y, x] = CLOUD
                    else:
                        mask[t, y, x] = CLEAR


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void water_land_filter(const float32[:, :, :] avg,
                            const float32[:, :, :] mndwi,
                            const float32[:, :, :] msavi,
                            uint8[:, :, :] result,
                            float32 thd,
                            float32 dwithd,
                            float32 cloudthd,
                            float32 landcloudthd,
                            float32 avithd,
                            float32 wtdthd) nogil:

    for t in range(avg.shape[0]):
        for y in range(avg.shape[1]):
            for x in range(avg.shape[2]):
                if result[t, y, x] == SHADOW and mndwi[t, y, x] > dwithd and avg[t, y, x] < thd:
                    result[t, y, x] = CLEAR

                if result[t, y, x] == CLOUD and avg[t, y, x] < cloudthd:
                    result[t, y, x] = CLEAR

                # bare ground, not cloud
                if result[t, y, x] == CLOUD and mndwi[t, y, x] < landcloudthd:
                    result[t, y, x] = CLEAR

                # water pixel, not shadow
                if result[t, y, x] == SHADOW and msavi[t, y, x] < avithd and mndwi[t, y, x] > wtdthd:
                    result[t, y, x] = CLEAR

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int find_pairs(const float32[:, :, :] avg,
                    const uint8[:, :, :] mask,
                    int y,
                    int x,
                    int N,
                    int[:] paidx,
                    float32[:] pamu) nogil:
    cdef int t1, t2, ss, cc
    cdef float32 mu

    ss = 0

    for t1 in range(avg.shape[0]):
        cc = 0
        mu = 0.

        for t2 in range(t1, avg.shape[0]):
            if mask[t2, y, x] == CLEAR:
                paidx[ss * N + cc] = t2
                mu = mu + avg[t2, y, x]
                cc = cc + 1
                if cc == N:
                    mu = mu / cc
                    pamu[ss] = 1.0 * mu / cc
                    break

        if cc == N:
            ss = ss + 1
        else:
            break

    return ss


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sort_indices(int[:] result, float32[:] values, int N) nogil:
    cdef int tmp, i, j

    for i in range(N):
        result[i] = i

    for i in range(N):
        for j in range(N - i - 1):
            if values[result[j]] > values[result[j + 1]]:
                tmp = result[j]
                result[j] = result[j + 1]
                result[j + 1] = tmp


@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void test_pair(const float32[:, :, :] avg,
                    const float32[:, :, :] mndwi,
                    uint8[:, :, :] result,
                    int y,
                    int x,
                    int N,
                    int[:] paidx,
                    int pp,
                    float32 cspkthd,
                    float32 sspkthd,
                    float32 cloudthd,
                    float32 shadowthd) nogil:

    cdef int i, k, lfb, rhb, pt, cc
    cdef float32 mid, mu, m1, m2, m3

    m1 = 0
    m3 = 0

    pt = pp * N

    mu = 0
    cc = 0

    lfb = paidx[pt]
    rhb = paidx[pt + N - 1]

    for i in range(pt, pt + N):
        k = paidx[i]
        if result[k, y, x] == CLEAR:
            mu += avg[k, y, x]
            cc += 1

    if cc == 0:
        return

    m2 = mu / cc

    # looking at left of the time series
    cc = 0
    mu = 0

    for i in reversed(range(lfb)):
        if result[i, y, x] == CLEAR:
            if avg[i, y, x] > shadowthd or mndwi[i, y, x] > 0:
                mu += avg[i, y, x]
                cc += 1

                if cc == N:
                    m1 = mu / cc
                    break

    if cc < N:
        cc = 0
        mu = 0

        for i in range(rhb + 1 + (3 * N) / 2, avg.shape[0]):
            if result[i, y, x] == CLEAR:
                if avg[i, y, x] > shadowthd or mndwi[i, y, x] > 0:
                    mu += avg[i, y, x]
                    cc += 1

                    if cc == N:
                        m1 = mu / cc
                        break

    # looking at right of the time series
    cc = 0
    mu = 0

    for i in range(rhb + 1, avg.shape[0]):
        if result[i, y, x] == CLEAR:
            if avg[i, y, x] > shadowthd or mndwi[i, y, x] > 0:
                mu += avg[i, y, x]
                cc += 1

                if cc == N:
                    m3 = mu / cc
                    break

    if cc < N:
        cc = 0
        mu = 0

        for i in reversed(range(lfb - (3 * N) / 2)):
            if result[i, y, x] == CLEAR:
                if avg[i, y, x] > shadowthd or mndwi[i, y, x] > 0:
                    mu += avg[i, y, x]
                    cc += 1

                    if cc == N:
                        m3 = mu / cc
                        break

    mid = (m1 + m3) / 2
    if m2 > mid:
        if (m2 - mid) / mid > cspkthd and m2 > cloudthd:
            for i in range(pt, pt + N):
                result[paidx[i], y, x] = CLOUD
    else:
        if (mid - m2) / m2 > sspkthd and m2 < shadowthd:
            for i in range(pt, pt + N):
                result[paidx[i], y, x] = SHADOW


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void spike_removal_worker(const float32[:, :, :] avg,
                               const float32[:, :, :] mndwi,
                               uint8[:, :, :] result,
                               int N,
                               int[:] paidx,
                               float32[:] pamu,
                               int[:] sts,
                               float32 cspkthd,
                               float32 sspkthd,
                               float32 cloudthd,
                               float32 shadowthd) nogil:
    cdef int y, x
    cdef int ss, i

    for y in range(avg.shape[1]):
        for x in range(avg.shape[2]):
            ss = find_pairs(avg, result, y, x, N, paidx, pamu)
            if ss <= 0:
                continue

            sort_indices(sts, pamu, ss)

            for i in range(ss):
                # brighter objects are more likely cloud
                # so apply correction in the descending order of brightness
                test_pair(avg, mndwi, result, y, x, N, paidx, sts[ss - i - 1],
                          cspkthd, sspkthd, cloudthd, shadowthd)


def spike_removal(const float32[:, :, :] avg,
                  const float32[:, :, :] mndwi,
                  uint8[:, :, :] result,
                  int N,
                  thresholds):
    cdef int[:] paidx
    cdef float32[:] pamu
    cdef int[:] sts

    paidx = numpy.full((N * avg.shape[0],), fill_value=-1, dtype=numpy.intc)
    pamu = numpy.full((avg.shape[0],), fill_value=-1., dtype=numpy.float32)
    sts = numpy.full((avg.shape[0],), fill_value=-1, dtype=numpy.intc)

    spike_removal_worker(avg, mndwi, result, N, paidx, pamu, sts,
                         <float32>thresholds['cspk'],
                         <float32>thresholds['sspk'],
                         <float32>thresholds['cloud'],
                         <float32>thresholds['shadow'])


def tsmask_temporal(const float32[:, :, :] avg,
                    const float32[:, :, :] mndwi,
                    const float32[:, :, :] msavi,
                    thresholds):

    cdef uint8[:, :, :] result
    cdef uint8[:, :, :] mask

    result = numpy.full((avg.shape[0], avg.shape[1], avg.shape[2]),
                        fill_value=NO_OBS, dtype=numpy.uint8)

    initial_guess(avg, result, <float32>thresholds['brightness'])

    # TODO enable this: spike_removal(avg, mndwi, result, 1, thresholds)
    spike_removal(avg, mndwi, result, 1, thresholds)
    spike_removal(avg, mndwi, result, 1, thresholds)
    spike_removal(avg, mndwi, result, 2, thresholds)
    spike_removal(avg, mndwi, result, 2, thresholds)
    spike_removal(avg, mndwi, result, 3, thresholds)

    water_land_filter(avg, mndwi, msavi, result,
                      <float32>thresholds['thd'],
                      <float32>thresholds['dwi'],
                      <float32>thresholds['cloud2'],
                      <float32>thresholds['land_cloud'],
                      <float32>thresholds['avi'],
                      <float32>thresholds['wtd'])
    return numpy.asarray(result)


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void spatial_filter(const uint8[:, :, :] mask, uint8[:, :, :] result) nogil:
    cdef int newlab, cc
    cdef int t, j, i, n, m, y, x

    for t in range(mask.shape[0]):
        for j in range(1, mask.shape[1] - 1):
            for i in range(1, mask.shape[2] - 1):
                if mask[t, j, i] == NO_OBS:
                    continue

                cc = 0
                newlab = 0

                for n in range(-1, 1 + 1):
                    for m in range(-1, 1 + 1):
                        if n == 0 and m == 0:
                            continue

                        y = j + n
                        x = i + m

                        if mask[t, y, x] == NO_OBS:
                            continue

                        if mask[t, j, i] == CLEAR:
                            if mask[t, y, x] == SHADOW or mask[t, y, x] == CLOUD:
                                newlab += mask[t, y, x]
                                cc = cc + 1
                        else:
                            if mask[t, y, x] == CLEAR:
                                cc = cc + 1

                if cc > 6:
                    if mask[t, j, i] == SHADOW or mask[t, j, i] == CLOUD:
                        result[t, j, i] = CLEAR
                    else:
                        # which ever is the majority (cloud/shadow)
                        result[t, j, i] = <uint8>round((1.0 * newlab) / cc)


def spatial_noise_filter(const uint8[:, :, :] mask):
    cdef uint8[:, :, :] result

    result = numpy.copy(mask)

    spatial_filter(mask, result)

    return numpy.asarray(result)
