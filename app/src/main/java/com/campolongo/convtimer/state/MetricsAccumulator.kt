package com.campolongo.convtimer.state

import java.util.concurrent.atomic.AtomicLong

data class MetricsSnapshot(
    val trt: Long = 0L,
    val wta: Long = 0L,
    val wtb: Long = 0L,
    val sta: Long = 0L,
    val stb: Long = 0L,
    val stm: Long = 0L,
) {
    val cta: Long get() = wta + sta
    val ctb: Long get() = wtb + stb
    val tct: Long get() = cta + stm + ctb
    val tst: Long get() = sta + stb + stm
    val bfst: Long get() = trt - tct
}

class MetricsAccumulator {
    private val _trt = AtomicLong(0L)
    private val _wta = AtomicLong(0L)
    private val _wtb = AtomicLong(0L)
    private val _sta = AtomicLong(0L)
    private val _stb = AtomicLong(0L)
    private val _stm = AtomicLong(0L)

    fun addTrt(dt: Long) { _trt.addAndGet(dt) }
    fun addWta(dt: Long) { _wta.addAndGet(dt) }
    fun addWtb(dt: Long) { _wtb.addAndGet(dt) }
    fun addSta(dt: Long) { _sta.addAndGet(dt) }
    fun addStb(dt: Long) { _stb.addAndGet(dt) }
    fun addStm(dt: Long) { _stm.addAndGet(dt) }

    fun snapshot(): MetricsSnapshot = MetricsSnapshot(
        trt = _trt.get(),
        wta = _wta.get(),
        wtb = _wtb.get(),
        sta = _sta.get(),
        stb = _stb.get(),
        stm = _stm.get(),
    )

    fun reset() {
        _trt.set(0L)
        _wta.set(0L)
        _wtb.set(0L)
        _sta.set(0L)
        _stb.set(0L)
        _stm.set(0L)
    }
}
