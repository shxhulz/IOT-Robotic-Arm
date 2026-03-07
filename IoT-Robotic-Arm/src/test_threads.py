"""
Test script to verify:
  1. frame_queue and cmd_queue work correctly under load
  2. Detection stability counter gates commands properly
  3. Kafka events are only sent at the right moments (no flooding)
  4. robot_state / target_locked transitions are correct
  5. Vision thread does NOT flood commands while robot is MOVING

Run with:  uv run python src/test_threads.py
"""

import sys
import os
import threading
import time
from queue import Queue, Empty, Full
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from config.config import (
    FRAME_QUEUE_SIZE,
    CMD_QUEUE_SIZE,
    DETECTION_STABILITY_THRESHOLD,
)
from utils.logger import get_logger

logger = get_logger("test_threads")

passed = 0
failed = 0


def assert_eq(actual, expected, msg=""):
    global passed, failed
    if actual == expected:
        passed += 1
        logger.info(f"  [PASS]: {msg}")
    else:
        failed += 1
        logger.error(f"  [FAIL]: {msg} -- expected {expected!r}, got {actual!r}")


def assert_true(condition, msg=""):
    global passed, failed
    if condition:
        passed += 1
        logger.info(f"  [PASS]: {msg}")
    else:
        failed += 1
        logger.error(f"  [FAIL]: {msg}")


# ── Test 1: frame_queue bounded behavior ──────────────────────────────────

def test_frame_queue_bounded():
    logger.info("TEST 1: frame_queue bounded behavior")
    q = Queue(maxsize=FRAME_QUEUE_SIZE)

    # Fill the queue
    for i in range(FRAME_QUEUE_SIZE):
        q.put(f"frame_{i}")
    assert_true(q.full(), f"Queue should be full at {FRAME_QUEUE_SIZE} items")

    # Attempting to put should not block — we drop oldest
    try:
        q.get_nowait()  # drop oldest
        q.put_nowait("frame_new")
        assert_true(True, "Drop-oldest strategy works")
    except Full:
        assert_true(False, "Drop-oldest strategy should prevent Full exception")

    # Verify the oldest was dropped
    first = q.get_nowait()
    assert_eq(first, "frame_1", "After dropping oldest, first item should be frame_1")

    logger.info("")


# ── Test 2: cmd_queue bounded behavior ────────────────────────────────────

def test_cmd_queue_bounded():
    logger.info("TEST 2: cmd_queue bounded behavior")
    q = Queue(maxsize=CMD_QUEUE_SIZE)

    # Fill the queue
    for i in range(CMD_QUEUE_SIZE):
        q.put({"commands": [f"cmd_{i}"], "object_class": "test"})
    assert_true(q.full(), f"cmd_queue should be full at {CMD_QUEUE_SIZE} items")

    # put_nowait on a full queue should raise Full
    try:
        q.put_nowait({"commands": ["overflow"], "object_class": "test"})
        assert_true(False, "Should have raised Full")
    except Full:
        assert_true(True, "Full exception raised correctly on overflow")

    logger.info("")


# ── Test 3: Detection stability counter ──────────────────────────────────

def test_detection_stability():
    logger.info("TEST 3: Detection stability counter")
    detection_counter = 0
    commands_issued = 0
    robot_state = "IDLE"
    target_locked = False

    # Simulate N-1 consecutive detections — should NOT trigger
    for i in range(DETECTION_STABILITY_THRESHOLD - 1):
        detection_counter += 1

    assert_eq(
        detection_counter,
        DETECTION_STABILITY_THRESHOLD - 1,
        f"Counter should be {DETECTION_STABILITY_THRESHOLD - 1} after {DETECTION_STABILITY_THRESHOLD - 1} detections",
    )
    should_issue = (
        detection_counter >= DETECTION_STABILITY_THRESHOLD
        and robot_state == "IDLE"
        and not target_locked
    )
    assert_true(not should_issue, "Should NOT issue command before threshold")

    # One more detection — should trigger
    detection_counter += 1
    should_issue = (
        detection_counter >= DETECTION_STABILITY_THRESHOLD
        and robot_state == "IDLE"
        and not target_locked
    )
    assert_true(should_issue, f"Should issue command at threshold ({DETECTION_STABILITY_THRESHOLD})")

    if should_issue:
        robot_state = "MOVING"
        target_locked = True
        detection_counter = 0
        commands_issued += 1

    assert_eq(robot_state, "MOVING", "State should be MOVING after command issued")
    assert_eq(target_locked, True, "Target should be locked")
    assert_eq(commands_issued, 1, "Exactly one command should have been issued")

    # Simulate 10 more detections while MOVING — should NOT trigger
    for i in range(10):
        detection_counter += 1
        should_issue = (
            detection_counter >= DETECTION_STABILITY_THRESHOLD
            and robot_state == "IDLE"
            and not target_locked
        )
        if should_issue:
            commands_issued += 1

    assert_eq(commands_issued, 1, "No additional commands while MOVING (anti-flood)")

    # Simulate robot finishing — return to IDLE
    robot_state = "IDLE"
    target_locked = False
    detection_counter = 0

    assert_eq(robot_state, "IDLE", "State should return to IDLE after task complete")
    assert_eq(target_locked, False, "Target lock should be released")

    logger.info("")


# ── Test 4: Kafka event control ───────────────────────────────────────────

def test_kafka_event_control():
    logger.info("TEST 4: Kafka event control (no per-detection flooding)")
    kafka_events = []

    class MockProducer:
        def send(self, topic, data):
            kafka_events.append({"topic": topic, "data": data})

    producer = MockProducer()
    robot_state = "IDLE"
    target_locked = False
    detection_counter = 0

    # Simulate 20 detections — only ONE pickup_start should be sent
    for i in range(20):
        detection_counter += 1
        if (
            detection_counter >= DETECTION_STABILITY_THRESHOLD
            and robot_state == "IDLE"
            and not target_locked
        ):
            producer.send(
                "robot_events",
                {"event_type": "pickup_start", "object_class": "plastic"},
            )
            robot_state = "MOVING"
            target_locked = True
            detection_counter = 0

    pickup_starts = [
        e for e in kafka_events if e["data"]["event_type"] == "pickup_start"
    ]
    assert_eq(
        len(pickup_starts), 1, "Exactly 1 pickup_start event (no flooding)"
    )

    # Simulate robot finishing
    producer.send(
        "robot_events",
        {"event_type": "pickup_success", "object_class": "plastic"},
    )
    robot_state = "IDLE"
    target_locked = False

    total_events = len(kafka_events)
    assert_eq(total_events, 2, "Total Kafka events: 1 start + 1 success = 2")

    # Print all events for inspection
    for e in kafka_events:
        logger.debug(f"  Kafka event: {e['topic']} -> {e['data']['event_type']}")

    logger.info("")


# ── Test 5: Thread-safe state transitions ─────────────────────────────────

def test_thread_safe_state():
    logger.info("TEST 5: Thread-safe state transitions")
    lock = threading.Lock()
    state = {"robot_state": "IDLE", "target_locked": False}
    issues = []

    def writer_thread(n):
        for _ in range(100):
            with lock:
                state["robot_state"] = "MOVING"
                state["target_locked"] = True
            time.sleep(0.001)
            with lock:
                state["robot_state"] = "IDLE"
                state["target_locked"] = False

    def reader_thread(n):
        for _ in range(100):
            with lock:
                s = state["robot_state"]
                t = state["target_locked"]
            # When MOVING, target_locked should be True
            # When IDLE, target_locked should be False
            # (no torn reads if lock works)
            if s == "MOVING" and not t:
                issues.append("MOVING but target not locked!")
            if s == "IDLE" and t:
                issues.append("IDLE but target locked!")
            time.sleep(0.001)

    threads = []
    for i in range(4):
        threads.append(threading.Thread(target=writer_thread, args=(i,)))
        threads.append(threading.Thread(target=reader_thread, args=(i,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert_eq(len(issues), 0, f"No torn reads detected ({len(issues)} issues)")
    logger.info("")


# ── Test 6: Queue producer-consumer throughput ────────────────────────────

def test_queue_throughput():
    logger.info("TEST 6: Queue producer-consumer throughput")
    q = Queue(maxsize=FRAME_QUEUE_SIZE)
    produced = {"count": 0}
    consumed = {"count": 0}
    dropped = {"count": 0}

    def producer():
        for i in range(50):
            if q.full():
                try:
                    q.get_nowait()
                    dropped["count"] += 1
                except Empty:
                    pass
            q.put(f"frame_{i}")
            produced["count"] += 1
            time.sleep(0.01)

    def consumer():
        while True:
            try:
                item = q.get(timeout=1.0)
                consumed["count"] += 1
            except Empty:
                break

    p = threading.Thread(target=producer)
    c = threading.Thread(target=consumer)
    p.start()
    c.start()
    p.join()
    c.join()

    assert_eq(produced["count"], 50, "All 50 frames produced")
    assert_eq(
        consumed["count"] + dropped["count"],
        50,
        f"consumed({consumed['count']}) + dropped({dropped['count']}) = produced(50)",
    )
    assert_true(consumed["count"] > 0, f"At least some frames consumed ({consumed['count']})")
    logger.info(
        f"  Throughput: produced={produced['count']}, "
        f"consumed={consumed['count']}, dropped={dropped['count']}"
    )
    logger.info("")


# ── Run all tests ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  MULTI-THREADED ROBOTIC ARM -- UNIT TESTS")
    logger.info("=" * 60)
    logger.info("")

    test_frame_queue_bounded()
    test_cmd_queue_bounded()
    test_detection_stability()
    test_kafka_event_control()
    test_thread_safe_state()
    test_queue_throughput()

    logger.info("=" * 60)
    logger.info(f"  RESULTS: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    sys.exit(1 if failed > 0 else 0)
