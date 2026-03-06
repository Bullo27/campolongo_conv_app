package com.campolongo.convtimer.state

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class MetricsSnapshotTest {

    private lateinit var acc: MetricsAccumulator

    @Before
    fun setUp() {
        acc = MetricsAccumulator()
    }

    @Test
    fun defaultSnapshotIsAllZeros() {
        val snap = acc.snapshot()
        assertEquals(0L, snap.trt)
        assertEquals(0L, snap.wta)
        assertEquals(0L, snap.wtb)
        assertEquals(0L, snap.sta)
        assertEquals(0L, snap.stb)
        assertEquals(0L, snap.stm)
        assertEquals(0L, snap.ovt)
        assertEquals(0L, snap.cta)
        assertEquals(0L, snap.ctb)
        assertEquals(0L, snap.tct)
        assertEquals(0L, snap.tst)
        assertEquals(0L, snap.bfst)
    }

    @Test
    fun addTrtAccumulatesCorrectly() {
        acc.addTrt(100)
        acc.addTrt(200)
        acc.addTrt(50)
        assertEquals(350L, acc.snapshot().trt)
    }

    @Test
    fun addWtaAccumulatesCorrectly() {
        acc.addWta(10)
        acc.addWta(20)
        assertEquals(30L, acc.snapshot().wta)
    }

    @Test
    fun addWtbAccumulatesCorrectly() {
        acc.addWtb(15)
        acc.addWtb(25)
        assertEquals(40L, acc.snapshot().wtb)
    }

    @Test
    fun addStaStbStmAccumulateCorrectly() {
        acc.addSta(5)
        acc.addStb(10)
        acc.addStm(15)
        val snap = acc.snapshot()
        assertEquals(5L, snap.sta)
        assertEquals(10L, snap.stb)
        assertEquals(15L, snap.stm)
    }

    @Test
    fun ctaIsWtaPlusSta() {
        acc.addWta(100)
        acc.addSta(50)
        assertEquals(150L, acc.snapshot().cta)
    }

    @Test
    fun ctbIsWtbPlusStb() {
        acc.addWtb(200)
        acc.addStb(80)
        assertEquals(280L, acc.snapshot().ctb)
    }

    @Test
    fun tctIsCtaPlusStmPlusCtb() {
        acc.addWta(100)
        acc.addSta(50)   // cta = 150
        acc.addWtb(200)
        acc.addStb(80)   // ctb = 280
        acc.addStm(30)
        assertEquals(150L + 30L + 280L, acc.snapshot().tct)
    }

    @Test
    fun tstIsStaPlusStbPlusStm() {
        acc.addSta(10)
        acc.addStb(20)
        acc.addStm(30)
        assertEquals(60L, acc.snapshot().tst)
    }

    @Test
    fun bfstIsTrtMinusTct() {
        acc.addTrt(1000)
        acc.addWta(300)
        acc.addWtb(200)
        acc.addSta(50)
        acc.addStb(50)
        acc.addStm(100)
        // tct = (300+50) + 100 + (200+50) = 700
        val snap = acc.snapshot()
        assertEquals(700L, snap.tct)
        assertEquals(300L, snap.bfst)
    }

    @Test
    fun addOvtAccumulatesCorrectly() {
        acc.addOvt(100)
        acc.addOvt(50)
        assertEquals(150L, acc.snapshot().ovt)
    }

    @Test
    fun overlapTrackedIndependently() {
        // During overlap, WTA and WTB both get credited for the same time.
        // OVT tracks the overlap amount separately.
        acc.addWta(500)  // 300 solo A + 200 overlap
        acc.addWtb(400)  // 200 solo B + 200 overlap
        acc.addOvt(200)
        val snap = acc.snapshot()
        assertEquals(200L, snap.ovt)
        // Unique speech time = WTA + WTB - OVT = 700
        assertEquals(700L, snap.wta + snap.wtb - snap.ovt)
    }

    @Test
    fun resetZeroesEverything() {
        acc.addTrt(1000)
        acc.addWta(100)
        acc.addWtb(200)
        acc.addSta(30)
        acc.addStb(40)
        acc.addStm(50)
        acc.addOvt(60)
        acc.reset()
        val snap = acc.snapshot()
        assertEquals(0L, snap.trt)
        assertEquals(0L, snap.wta)
        assertEquals(0L, snap.wtb)
        assertEquals(0L, snap.sta)
        assertEquals(0L, snap.stb)
        assertEquals(0L, snap.stm)
        assertEquals(0L, snap.ovt)
    }

    @Test
    fun snapshotIsImmutable() {
        acc.addWta(100)
        val snap1 = acc.snapshot()
        acc.addWta(200)
        val snap2 = acc.snapshot()
        assertEquals(100L, snap1.wta)
        assertEquals(300L, snap2.wta)
    }
}
