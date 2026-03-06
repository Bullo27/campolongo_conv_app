package com.campolongo.convtimer.state

data class MetricsSnapshot(
    val trt: Long = 0L,
    val wta: Long = 0L,
    val wtb: Long = 0L,
    val sta: Long = 0L,
    val stb: Long = 0L,
    val stm: Long = 0L,
    val ovt: Long = 0L,
) {
    val cta: Long get() = wta + sta
    val ctb: Long get() = wtb + stb
    val tct: Long get() = cta + stm + ctb
    val tst: Long get() = sta + stb + stm
    val bfst: Long get() = trt - tct
}

/**
 * Accumulates conversation timing metrics.
 * All access must be from a single thread (Dispatchers.Main via viewModelScope).
 */
class MetricsAccumulator {
    private var _trt = 0L
    private var _wta = 0L
    private var _wtb = 0L
    private var _sta = 0L
    private var _stb = 0L
    private var _stm = 0L
    private var _ovt = 0L

    fun addTrt(dt: Long) { _trt += dt }
    fun addWta(dt: Long) { _wta += dt }
    fun addWtb(dt: Long) { _wtb += dt }
    fun addSta(dt: Long) { _sta += dt }
    fun addStb(dt: Long) { _stb += dt }
    fun addStm(dt: Long) { _stm += dt }
    fun addOvt(dt: Long) { _ovt += dt }

    fun snapshot(): MetricsSnapshot = MetricsSnapshot(
        trt = _trt,
        wta = _wta,
        wtb = _wtb,
        sta = _sta,
        stb = _stb,
        stm = _stm,
        ovt = _ovt,
    )

    fun reset() {
        _trt = 0L
        _wta = 0L
        _wtb = 0L
        _sta = 0L
        _stb = 0L
        _stm = 0L
        _ovt = 0L
    }
}
