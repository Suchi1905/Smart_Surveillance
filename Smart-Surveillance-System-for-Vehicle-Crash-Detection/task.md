# Smart Surveillance Enhancement Implementation

## Phase 1: Foundation
- [x] **Remove Privacy Feature**
  - [x] Remove `anonymize_frame` import from `detection.py`
  - [x] Remove privacy call in `_trigger_alert` method (line 624)
  - [x] Add config flag to optionally enable/disable anonymization
  - [x] Update tests

- [/] **Indian Traffic Adaptation**
  - [ ] Create `traffic_profiles.py` with US/Indian configurations
  - [ ] Add adaptive thresholds based on traffic density
  - [ ] Modify `behavior.py` threshold constants
  - [ ] Modify `collision.py` TTC/distance constants
  - [ ] Add vehicle type classification support

- [ ] **Risk Scoring System**
  - [ ] Create `risk_scorer.py` service
  - [ ] Implement composite risk formula
  - [ ] Integrate with behavior and collision services
  - [ ] Add risk visualization overlay

## Phase 2: Prediction
- [ ] **Pre-Accident Behavior Enhancement**
  - [ ] Add time-series pattern analysis
  - [ ] Implement behavior sequence modeling
  - [ ] Add confidence decay over time
  - [ ] Extend behavior types

- [ ] **Trajectory Prediction (LSTM)**
  - [ ] Create `trajectory_predictor.py` service
  - [ ] Implement Kalman filter prediction (short-term)
  - [ ] Add optional LSTM model for long-term prediction
  - [ ] Integrate with tracker and collision services

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
- [ ] **YOLO + Vision Transformer Hybrid**
  - [ ] Research MobileViT/EfficientViT integration
  - [ ] Implement cross-attention fusion layer
  - [ ] Train hybrid model
  - [ ] Benchmark performance
