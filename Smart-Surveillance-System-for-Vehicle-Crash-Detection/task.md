# Smart Surveillance Enhancement Implementation

## Phase 1: Foundation
- [x] **Remove Privacy Feature**
  - [x] Remove `anonymize_frame` import from `detection.py`
  - [x] Remove privacy call in `_trigger_alert` method (line 624)
  - [x] Add config flag to optionally enable/disable anonymization
  - [x] Update tests

- [/] **Indian Traffic Adaptation**
  - [x] Create `traffic_profiles.py` with US/Indian configurations
  - [x] Add adaptive thresholds based on traffic density
  - [x] Modify `behavior.py` threshold constants
  - [x] Modify `collision.py` TTC/distance constants
  - [x] **Debug Telegram Alerts**
    - [x] Verify API connectivity (`scripts/test_telegram_simple.py`)
    - [x] Verify Service Logic (`scripts/test_telegram_service.py`)
    - [x] Add verbose logging for skipped alerts
    - [x] Make alert severity configurable (`ALERT_SEVERITY_LEVELS`)
  - [ ] Add vehicle type classification support

- [/] **Risk Scoring System**
  - [x] Create `risk_scorer.py` service
  - [x] Implement composite risk formula
  - [x] Integrate with behavior and collision services
  - [x] Add risk visualization overlay

## Phase 2: Prediction
- [/] **Pre-Accident Behavior Enhancement**
  - [x] Add time-series pattern analysis
  - [x] Implement behavior sequence modeling
  - [x] Add confidence decay over time
  - [x] Extend behavior types

- [/] **Trajectory Prediction (LSTM)**
  - [x] Create `trajectory_predictor.py` service
  - [x] Implement Kalman filter prediction (short-term)
  - [x] Add optional LSTM model for long-term prediction
  - [x] Integrate with tracker and collision services

## Phase 3: Hardware Integration
- [ ] **Raspberry Pi Warning Controller**
  - [ ] Create `hardware/warning_controller.py`
  - [ ] Define GPIO interface for LED/speaker
  - [ ] Implement directional warning logic
  - [ ] Add hardware abstraction layer

- [ ] **Speed Hardware Integration**
  - [ ] Create `hardware/speed_hardware.py`
  - [ ] Add radar/LIDAR sensor interfaces
  - [ ] Implement sensor fusion with camera estimation

- [ ] **Road Sign Recognition**
  - [ ] Add road sign classes to YOLO training data
  - [ ] Update detection pipeline for sign detection
  - [ ] Implement speed limit enforcement logic

## Phase 4: Advanced AI (Future)
- [x] **YOLO + Vision Transformer Hybrid**
  - [x] Create training configuration (`hybrid_config.py`)
  - [x] Implement dataset loaders (`dataset_loader.py`)
  - [x] Phase A: Train YOLO backbone on KITTI (mAP50=89.0%, 5.56hrs)
  - [x] Phase B: Train ViT classifier on CCD (Val Acc=100%, ~2hrs)
  - [x] Phase C: Train fusion layer (Val Acc=100%, ~1.3hrs)
  - [x] Phase D: End-to-end fine-tuning (Val Acc=100%, ~42min)
  - [x] Integration into detection service (`hybrid_classifier.py`)


## Phase 5: Testing
- [/] **Integration Testing**
  - [/] Test Infrastructure
    - [x] Update `conftest.py` with comprehensive fixtures
    - [x] Add mock DetectionService, Telegram, frame sequences, timer
  - [/] End-to-End Pipeline Tests (`test_e2e_pipeline.py`)
    - [x] Pipeline initialization tests
    - [x] Frame processing tests
    - [x] Severity triage flow tests (early confidence classification)
    - [x] Alert triggering tests (cooldown, fallback)
    - [x] Triage reset on new stream
  - [/] Alert & Telegram Integration (`test_alert_integration.py`)
    - [x] Telegram service init & disabled handling
    - [x] Alert dispatch with mocked HTTP
    - [x] Alert callback invocation
    - [x] send_telegram_alert wrapper tests
  - [/] Detection Service Unit Tests (`test_detection_service.py`)
    - [x] Service initialization & state
    - [x] IoU calculation
    - [x] Alert cooldown timing
    - [x] Frame encoding & error frames
    - [x] Severity levels configuration
  - [/] Performance Benchmarks (`test_performance.py`)
    - [x] Frame processing latency (<500ms)
    - [x] Triage throughput (>100/s)
    - [x] Encoding speed (<10ms)
    - [x] Severity calculation speed
  - [/] Updated Severity Triage Tests (`test_severity_triage.py`)
    - [x] Early confidence classification (7 tests)
    - [x] Two-frame motion analysis (2 tests)
  - [/] Scenario Test Runner (`tests/scenarios/run_scenario.py`)
    - [x] CLI script for real-world video testing
    - [ ] Run scenario on crash video
    - [ ] Run scenario on normal traffic video
  - [x] Run full test suite and verify all pass
