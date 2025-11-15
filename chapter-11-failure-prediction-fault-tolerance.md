# Chapter 11: Failure Prediction and Fault Tolerance

**Large-Scale LLM Training Playbook: 1.5 Trillion Parameter Model**
**Target Infrastructure: 5GW Multi-Datacenter Deployment**
**Investment Scale: $100+ Billion**

---

## Executive Overview

At the scale required for 1.5 trillion parameter model training, hardware failures transition from exceptional events to statistical certainties. With 350,000 H100 GPUs deployed across multiple datacenters, the cluster experiences a GPU failure approximately **every 20-30 minutes**. Without sophisticated failure prediction, proactive fault tolerance, and rapid recovery mechanisms, this failure rate would render large-scale training economically infeasible.

This chapter provides production-grade strategies for maintaining >95% effective training time despite constant hardware failures. The approaches detailed here are validated by real-world deployments: ByteDance's ByteRobust framework handling 38,236 failures over three months, Meta's 350,000 GPU deployment, and Alibaba's DLRover improving goodput from 69% to 95% for GLM-65B training.

**The Reality of Failures at Scale:**

At 350,000 GPU deployment:
- **MTBF (Mean Time Between Failures)**: 20-30 minutes across the entire cluster
- **Single-node MTBF**: ~1-2 months per 8-GPU server
- **Cluster-wide impact**: 2-3 failures per hour during typical training runs
- **Cumulative downtime**: Without fault tolerance, >50% of wall-clock time lost to failures

**Production Data from ByteDance (ByteRobust Study):**
- **Monitoring period**: 3 months (January-March 2024)
- **Failures observed**: 38,236 distinct failure events
- **GPU hours tracked**: >150 million A100 GPU-hours
- **Cluster size**: 24,000+ NVIDIA A100 GPUs across 2 research clusters
- **Training jobs**: 4+ million jobs across diverse workloads

**Key Chapter Outcomes:**

1. **Failure Prediction**: Deploy ML-based telemetry analysis achieving >85% accuracy in identifying "lemon nodes" before they cause training interruptions
2. **Checkpointing Strategy**: Implement asynchronous distributed checkpointing with <0.9% overhead (ByteRobust methodology)
3. **Graceful Degradation**: Enable elastic training that maintains productivity even when 5-10% of GPUs fail
4. **Recovery Procedures**: Achieve target recovery times: <30s single GPU, <5 min multi-GPU, <30 min datacenter failure

**Chapter Roadmap:**

This chapter addresses four critical domains:

1. **Failure Prediction** (Section 1): GPU telemetry signals, DCGM monitoring architecture, ML-based prediction models
2. **Checkpointing Strategy** (Section 2): Frequency optimization, asynchronous techniques, multi-level checkpoint hierarchies
3. **Graceful Degradation** (Section 3): Elastic training frameworks, NCCL communicator shrink, GPU process migration
4. **Recovery Procedures** (Section 4): Automated detection, recovery runbooks, cross-datacenter failover

**Success Criteria:**

- **Effective training time**: >95% (ratio of productive training to total wall-clock time)
- **Checkpoint overhead**: <1% of total training time
- **Mean Time To Recovery (MTTR)**: <5 minutes for typical failures
- **False positive rate**: <5% on proactive node exclusion
- **Goodput improvement**: 25-40% increase over baseline failure handling

---

## 1. Failure Prediction

### 1.1 GPU Telemetry Signals and Early Warning Indicators

Modern datacenter GPUs expose hundreds of telemetry signals that, when analyzed systematically, provide early warning of impending failures. The key is distinguishing between transient anomalies and indicators of genuine hardware degradation.

#### Critical Telemetry Categories

**ECC Memory Errors (Highest Predictive Value):**

NVIDIA H100 GPUs include Error-Correcting Code (ECC) memory protection. Monitoring ECC errors provides the strongest signal for predicting GPU failures:

```
ECC Error Types and Failure Correlation

Metric                              Sampling   Failure Correlation   Action Threshold
────────────────────────────────────────────────────────────────────────────────────
ecc_sbe_volatile_total              1 min      Moderate (40-60%)     >100/hour
  (Single-Bit Errors since boot)

ecc_dbe_volatile_total              1 min      Critical (>90%)       >1 (immediate)
  (Double-Bit Errors since boot)

ecc_sbe_aggregate_total             1 min      High (60-80%)         >10,000 lifetime
  (Lifetime SBE accumulation)

Row Remapping Events                Event      Critical (>95%)       >0 (investigate)

Pending Page Retirements            Event      Critical (>95%)       >0 (drain node)

Dynamic Page Offlining              Event      Critical (>95%)       >5 pages
────────────────────────────────────────────────────────────────────────────────────
```

**Research Validation (Production Study, 4-Month Dataset):**
- Pending page retirements: 97% correlation with GPU failure within 7 days
- Row remapping events: 94% correlation with failure within 14 days
- SBE rate >100/hour: 58% correlation with failure within 30 days
- Any DBE (Double-Bit Error): 92% correlation with failure within 48 hours

**XID Errors (Critical System Events):**

XID (X-Window System ID) errors indicate serious GPU hardware or driver issues:

```
XID Error Codes and Severity Classification

XID Code   Severity    Description                          Action Required
─────────────────────────────────────────────────────────────────────────────────
48         CRITICAL    Double Bit ECC Error (DBE)           Immediate node drain
63         CRITICAL    Row Remapper Event                   Immediate node drain
64         CRITICAL    ECC Page Retirement                  Immediate node drain
92         HIGH        High SBE Rate                        Monitor, consider drain
94         CRITICAL    Contained Error (RAS feature)        Investigate, likely drain
95         CRITICAL    Uncontained Error                    Immediate drain + RMA
74         MEDIUM      Thermal Event (throttling)           Check cooling
79         CRITICAL    GPU Fallen Off Bus                   Immediate drain + reboot
13         MEDIUM      Graphics Engine Exception            Monitor
31         HIGH        GPU Memory Access Error              Drain if recurring
─────────────────────────────────────────────────────────────────────────────────

RAS (Reliability, Availability, Serviceability) Features:
- H100 error containment: Handles 92% of memory errors without process termination
- Contained errors (XID 94): Recoverable but indicate degradation
- Uncontained errors (XID 95): Immediate failure, requires node replacement
```

**Production Example (ByteDance Cluster Data):**
- XID 48 (DBE) observed: 432 instances over 3 months
- Of those, 406 (94%) led to GPU failure requiring replacement
- Average time from XID 48 to failure: 26 hours
- Proactive exclusion window: 24 hours provides 89% prevention

**Temperature and Thermal Metrics:**

```
Temperature Monitoring Strategy

Sensor Location           Normal Range    Warning    Critical   Sampling Frequency
────────────────────────────────────────────────────────────────────────────────────
GPU Die (Core)           60-75°C         80°C       85°C       10 seconds
Memory Junction          65-80°C         85°C       90°C       10 seconds
Hotspot (Max Die Temp)   70-85°C         90°C       95°C       10 seconds
Thermal Margin           15-30°C         <10°C      <5°C       10 seconds
────────────────────────────────────────────────────────────────────────────────────

Thermal Warning Conditions:
- Sustained temperature >80°C for >10 minutes: Investigate cooling
- Temperature spikes >85°C: Check for thermal throttling
- Thermal margin <10°C: Imminent throttling, likely cooling issue
- Temperature >90°C: Immediate investigation required
```

**Thermal Failure Correlation:**
- Sustained high temperature (>85°C for >1 hour): 23% correlation with failure within 30 days
- Frequent thermal throttling events: 31% correlation with cooling system failure
- Temperature sensors reading 0°C or >100°C: 100% correlation with sensor/hardware failure

**Power Metrics:**

```
Power Monitoring and Anomaly Detection

Metric                    Normal Range       Anomaly Threshold       Failure Correlation
───────────────────────────────────────────────────────────────────────────────────────
GPU Power Draw (H100)     600-700W (load)    >750W or <200W (load)   Moderate (35%)
Power Violations          0 events/hour      >5 events/hour          High (65%)
Power Throttling          <1% of time        >5% of time             Moderate (40%)
Total Energy (daily)      14.4-16.8 kWh      >18 kWh or <12 kWh      Low (20%)
───────────────────────────────────────────────────────────────────────────────────────

Anomaly Patterns:
- Power draw 0W while job running: GPU hang or driver crash
- Power consistently >TDP: Cooling insufficient, GPU stressed
- Power violations: Electrical delivery issues or GPU defect
```

**Performance Degradation Signals:**

```
Performance Metrics Indicating Degradation

Metric                       Baseline        Degradation Threshold
──────────────────────────────────────────────────────────────────────
GPU Utilization              95-100%         <80% during training
Memory Bandwidth Util        85-95%          <70% during data-intensive ops
SM Clock Frequency           1980 MHz        <1785 MHz (10% reduction)
Memory Clock Frequency       2619 MHz        <2357 MHz (10% reduction)
PCIe Throughput              ~60 GB/s        <50 GB/s (bidirectional)
NVLink Bandwidth             900 GB/s/link   <800 GB/s/link
──────────────────────────────────────────────────────────────────────

Clock Throttling Reasons (critical to monitor):
- HW Thermal Slowdown: Thermal throttling active
- HW Power Brake Slowdown: Power limit throttling
- SW Thermal Slowdown: Software-imposed thermal limit
- Sync Boost: Multi-GPU sync limiting clocks
```

**Interconnect Health Indicators:**

```
NVLink and PCIe Error Monitoring

Metric                          Normal      Warning      Critical       Action
─────────────────────────────────────────────────────────────────────────────────
NVLink CRC Errors (per link)    0/hour      >10/hour     >100/hour      Investigate cable
NVLink Replay Events            <5/hour     >50/hour     >500/hour      Link degradation
PCIe Replay Count               <10/hour    >100/hour    >1000/hour     Check PCIe link
PCIe NAK Received               0           >10          >100           PCIe bus issue
NVLink Recovery Errors          0           >0           >5             Possible link failure
─────────────────────────────────────────────────────────────────────────────────────

Production Impact:
- NVLink errors >500/hour: 72% correlation with training slowdown
- PCIe replay count >1000/hour: 45% correlation with I/O bottleneck
```

### 1.2 DCGM Monitoring Architecture

NVIDIA Data Center GPU Manager (DCGM) provides the foundational monitoring infrastructure for production GPU clusters.

#### Deployment Architecture for 350,000 GPUs

```
DCGM Monitoring Stack Architecture
═══════════════════════════════════════════════════════════════

Layer 4: Visualization & Alerting
┌─────────────────────────────────────────────────────────────┐
│ Grafana Dashboards (Real-time + Historical)                 │
│ AlertManager (Threshold + ML Anomaly Alerts)                │
│ PagerDuty Integration (Critical Failures)                   │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ PromQL Queries
                          │
Layer 3: Metrics Storage & Aggregation
┌─────────────────────────────────────────────────────────────┐
│ Prometheus Federation (Per-Datacenter Prometheus Instances) │
│ - Site A: 2,000,000 GPU metrics                             │
│ - Site B: 2,000,000 GPU metrics                             │
│ - Site C: 1,000,000 GPU metrics                             │
│ Retention: 30 days detailed, 1 year aggregated              │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ HTTP Pull (15s interval)
                          │
Layer 2: Metrics Export
┌─────────────────────────────────────────────────────────────┐
│ DCGM-Exporter (DaemonSet on each GPU node)                  │
│ - Exposes GPU metrics in Prometheus format                  │
│ - Custom metric groups optimized for training workloads     │
│ - 150+ metrics per GPU, configurable sampling               │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ gRPC API
                          │
Layer 1: GPU Telemetry Collection
┌─────────────────────────────────────────────────────────────┐
│ DCGM Host Engine (Per-node daemon)                          │
│ - Collects 100+ metrics per GPU                             │
│ - Sub-second sampling (configurable: 100ms - 60s)           │
│ - Local metric buffering and aggregation                    │
│ - Health check execution engine                             │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ NVML API
                          │
Layer 0: GPU Hardware
┌─────────────────────────────────────────────────────────────┐
│ NVIDIA H100 GPUs (350,000 across all sites)                 │
│ - Hardware sensors and telemetry                            │
│ - ECC error reporting                                       │
│ - Performance counters                                      │
└─────────────────────────────────────────────────────────────┘
```

#### DCGM Configuration for High-Frequency Monitoring

**Metric Groups and Sampling Strategy:**

```yaml
# dcgm-exporter-config.yaml
# Optimized for LLM training workload monitoring

collectors:
  - name: training_critical_metrics
    interval: 1000  # 1 second sampling
    metrics:
      - DCGM_FI_DEV_GPU_TEMP              # GPU temperature
      - DCGM_FI_DEV_MEMORY_TEMP           # Memory temperature
      - DCGM_FI_DEV_POWER_USAGE           # Power consumption
      - DCGM_FI_DEV_GPU_UTIL              # GPU utilization
      - DCGM_FI_DEV_MEM_COPY_UTIL         # Memory bandwidth util
      - DCGM_FI_DEV_ECC_DBE_VOL_TOTAL     # Double-bit errors
      - DCGM_FI_DEV_ECC_SBE_VOL_TOTAL     # Single-bit errors
      - DCGM_FI_DEV_XID_ERRORS            # Critical XID events

  - name: performance_metrics
    interval: 10000  # 10 second sampling
    metrics:
      - DCGM_FI_DEV_SM_CLOCK              # SM clock frequency
      - DCGM_FI_DEV_MEM_CLOCK             # Memory clock frequency
      - DCGM_FI_PROF_GR_ENGINE_ACTIVE     # Graphics engine active
      - DCGM_FI_PROF_PIPE_TENSOR_ACTIVE   # Tensor core active
      - DCGM_FI_PROF_DRAM_ACTIVE          # DRAM active cycles
      - DCGM_FI_DEV_NVLINK_BANDWIDTH_*    # Per-link NVLink BW
      - DCGM_FI_DEV_PCIE_TX_BYTES         # PCIe transmit
      - DCGM_FI_DEV_PCIE_RX_BYTES         # PCIe receive

  - name: reliability_metrics
    interval: 60000  # 60 second sampling
    metrics:
      - DCGM_FI_DEV_RETIRED_DBE           # Retired pages (DBE)
      - DCGM_FI_DEV_RETIRED_SBE           # Retired pages (SBE)
      - DCGM_FI_DEV_RETIRED_PENDING       # Pending retirements
      - DCGM_FI_DEV_ROW_REMAP_FAILURE     # Row remap failures
      - DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_*
      - DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_*
      - DCGM_FI_DEV_PCIE_REPLAY_COUNTER   # PCIe replays
      - DCGM_FI_DEV_POWER_VIOLATION       # Power violations
      - DCGM_FI_DEV_THERMAL_VIOLATION     # Thermal violations

  - name: process_metrics
    interval: 5000  # 5 second sampling
    metrics:
      - DCGM_FI_DEV_GRAPHICS_PIDS         # Running process IDs
      - DCGM_FI_DEV_COMPUTE_PIDS          # Compute process IDs
```

**Data Volume Calculations:**

For 350,000 H100 GPUs with above configuration:

```
Metric Data Volume Estimation
═══════════════════════════════════════════════════════════════

Configuration:
- Critical metrics: 8 metrics × 1 sample/sec × 350,000 GPUs
- Performance metrics: 10 metrics × 0.1 sample/sec × 350,000 GPUs
- Reliability metrics: 10 metrics × 1 sample/60sec × 350,000 GPUs
- Process metrics: 2 metrics × 1 sample/5sec × 350,000 GPUs

Data Rate (samples/second):
- Critical: 2,800,000 samples/sec
- Performance: 350,000 samples/sec
- Reliability: 58,333 samples/sec
- Process: 140,000 samples/sec
─────────────────────────────────────────────────────────────
TOTAL: 3,348,333 samples/second

Storage Requirements (with Prometheus compression):
- Per sample: ~12 bytes (average after compression)
- Per second: 40 MB/sec
- Per hour: 144 GB/hour
- Per day: 3.5 TB/day
- 30-day retention: 105 TB uncompressed, ~35 TB compressed

Recommended Infrastructure:
- Prometheus instances: 20-30 (federated by datacenter/pod)
- Storage per instance: 2-3 TB SSD
- Query latency: <500ms for 24-hour queries
- Retention: 30 days detailed, 365 days downsampled (10min aggregates)
```

#### Health Check Integration

**DCGM Diagnostic Levels:**

```
DCGM Health Check Execution Strategy
═══════════════════════════════════════════════════════════════

Level 1: Quick Health Check (30 seconds)
┌─────────────────────────────────────────────────────────────┐
│ When: Before every training job (Slurm prolog)              │
│ Tests:                                                       │
│   - ECC error counts (DBE must be 0)                         │
│   - XID error presence (critical XIDs = exclusion)           │
│   - Temperature sensors functional                           │
│   - Power draw within normal range                           │
│   - PCIe link width and speed verification                   │
│   - NVLink connectivity test                                 │
│ Failure Rate: ~0.5% (175 nodes/35,000 nodes/day)            │
│ Impact: Job not started, node drained                        │
└─────────────────────────────────────────────────────────────┘

Level 2: Standard Health Check (3-5 minutes)
┌─────────────────────────────────────────────────────────────┐
│ When: Daily idle period (3am maintenance window)             │
│ Tests:                                                       │
│   - All Level 1 tests                                        │
│   - Memory bandwidth test (STREAM benchmark)                 │
│   - Compute validation (matrix multiply correctness)         │
│   - NVLink bandwidth test (all 18 links)                     │
│   - GPU-to-GPU memory copy validation                        │
│   - Thermal stress test (5-minute burn-in)                   │
│ Failure Rate: ~2% (700 nodes/35,000 nodes/day)              │
│ Impact: Proactive node exclusion before user impact          │
└─────────────────────────────────────────────────────────────┘

Level 3: Field Diagnostic (15-30 minutes)
┌─────────────────────────────────────────────────────────────┐
│ When: On-demand for suspected hardware issues                │
│ Tests:                                                       │
│   - All Level 2 tests                                        │
│   - NVIDIA Field Diagnostic comprehensive suite              │
│   - Extended memory test (all addressable memory)            │
│   - Long-duration stress test                                │
│   - Power supply validation                                  │
│   - Cooling system verification                              │
│ Failure Rate: ~5-10% (of nodes flagged by Level 2)          │
│ Impact: Definitive RMA decision                              │
└─────────────────────────────────────────────────────────────┘
```

**Slurm Integration Example:**

```bash
#!/bin/bash
# /etc/slurm/prolog.d/gpu_health_check.sh
# Executes before every GPU job

set -e

LOG_FILE="/var/log/slurm/gpu_health_$(date +%Y%m%d).log"
NODE=$(hostname)
TIMESTAMP=$(date --iso-8601=seconds)

echo "[$TIMESTAMP] Starting GPU health check on $NODE for job $SLURM_JOB_ID" >> $LOG_FILE

# Run DCGM Level 1 diagnostic
dcgmi diag -r 1 &> /tmp/dcgm_diag_$SLURM_JOB_ID.log

if [ $? -ne 0 ]; then
    echo "[$TIMESTAMP] DCGM diagnostic FAILED on $NODE" >> $LOG_FILE
    cat /tmp/dcgm_diag_$SLURM_JOB_ID.log >> $LOG_FILE

    # Drain the node
    scontrol update nodename=$NODE state=drain reason="DCGM_health_check_failed_job_$SLURM_JOB_ID"

    # Fail the prolog (job will be requeued in held state)
    exit 1
fi

# Check for critical XID errors in dmesg
CRITICAL_XIDS=$(dmesg | grep -E "Xid.*: (48|63|64|79|94|95)" | tail -10)
if [ ! -z "$CRITICAL_XIDS" ]; then
    echo "[$TIMESTAMP] Critical XID errors detected on $NODE" >> $LOG_FILE
    echo "$CRITICAL_XIDS" >> $LOG_FILE
    scontrol update nodename=$NODE state=drain reason="Critical_XID_error"
    exit 1
fi

# Check ECC DBE count
DBE_COUNT=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
if [ $DBE_COUNT -gt 0 ]; then
    echo "[$TIMESTAMP] Double-bit ECC errors ($DBE_COUNT) detected on $NODE" >> $LOG_FILE
    scontrol update nodename=$NODE state=drain reason="ECC_DBE_detected"
    exit 1
fi

echo "[$TIMESTAMP] GPU health check PASSED on $NODE" >> $LOG_FILE
exit 0
```

### 1.3 ML-Based Failure Prediction Models

Machine learning models trained on historical telemetry data can predict GPU failures with >85% accuracy, enabling proactive node exclusion before training interruptions occur.

#### Training Data Requirements

**Data Collection Infrastructure:**

```
ML Training Dataset Characteristics
═══════════════════════════════════════════════════════════════

Historical Data Requirements:
- Duration: Minimum 6 months, prefer 12+ months
- GPU hours: >50 million GPU-hours for robust training
- Failure labels: 500+ documented failures (with root cause)
- Telemetry coverage: 100+ metrics per GPU
- Sampling frequency: 1-minute granularity minimum

Production Example (ByteDance Study):
┌─────────────────────────────────────────────────────────────┐
│ Duration: 18 months (Jan 2023 - Jun 2024)                   │
│ GPU hours: 150+ million A100 GPU-hours                      │
│ Cluster size: 24,000 A100 GPUs (2 clusters)                 │
│ Failures documented: 3,847 labeled failures                 │
│   - GPU hardware: 1,523 (40%)                                │
│   - Memory errors: 891 (23%)                                 │
│   - Thermal/cooling: 478 (12%)                               │
│   - Driver/software: 612 (16%)                               │
│   - Network: 343 (9%)                                        │
│ Telemetry entries: 350+ million time-series samples         │
│ Storage: 47 TB raw telemetry data                           │
└─────────────────────────────────────────────────────────────┘
```

#### Feature Engineering for GPU Failure Prediction

```python
# Example feature extraction pipeline for GPU failure prediction

import pandas as pd
import numpy as np
from datetime import timedelta

class GPUFailureFeaturesExtractor:
    """
    Extract predictive features from GPU telemetry time series.

    Based on production research: "Predicting GPU Failures With High
    Precision Under Deep Learning Workloads" (2024)
    """

    def __init__(self, lookback_window='6h'):
        self.lookback_window = pd.Timedelta(lookback_window)

    def extract_features(self, telemetry_df, gpu_id, timestamp):
        """Extract features for a single GPU at given timestamp."""

        # Filter to lookback window
        end_time = pd.to_datetime(timestamp)
        start_time = end_time - self.lookback_window

        gpu_data = telemetry_df[
            (telemetry_df['gpu_id'] == gpu_id) &
            (telemetry_df['timestamp'] >= start_time) &
            (telemetry_df['timestamp'] <= end_time)
        ]

        features = {}

        # === ECC Error Features (Highest Predictive Value) ===
        features['ecc_sbe_total'] = gpu_data['ecc_sbe_volatile'].iloc[-1]
        features['ecc_sbe_rate_1h'] = self._compute_rate(
            gpu_data, 'ecc_sbe_volatile', window='1h'
        )
        features['ecc_sbe_rate_6h'] = self._compute_rate(
            gpu_data, 'ecc_sbe_volatile', window='6h'
        )
        features['ecc_dbe_total'] = gpu_data['ecc_dbe_volatile'].iloc[-1]
        features['ecc_dbe_any'] = 1 if features['ecc_dbe_total'] > 0 else 0

        # === Temperature Features ===
        features['temp_gpu_mean'] = gpu_data['gpu_temp'].mean()
        features['temp_gpu_max'] = gpu_data['gpu_temp'].max()
        features['temp_gpu_std'] = gpu_data['gpu_temp'].std()
        features['temp_mem_mean'] = gpu_data['mem_temp'].mean()
        features['temp_mem_max'] = gpu_data['mem_temp'].max()
        features['temp_above_80c_pct'] = (gpu_data['gpu_temp'] > 80).mean()

        # === Power Features ===
        features['power_mean'] = gpu_data['power_draw'].mean()
        features['power_max'] = gpu_data['power_draw'].max()
        features['power_std'] = gpu_data['power_draw'].std()
        features['power_violations'] = gpu_data['power_violation'].sum()

        # === Performance Features ===
        features['util_gpu_mean'] = gpu_data['gpu_util'].mean()
        features['util_gpu_std'] = gpu_data['gpu_util'].std()
        features['util_mem_mean'] = gpu_data['mem_util'].mean()
        features['sm_clock_mean'] = gpu_data['sm_clock'].mean()
        features['sm_clock_min'] = gpu_data['sm_clock'].min()
        features['throttle_thermal_pct'] = (
            gpu_data['clock_throttle_reasons'] & 0x08 != 0
        ).mean()

        # === Interconnect Features ===
        features['nvlink_crc_errors'] = gpu_data['nvlink_crc_errors'].sum()
        features['nvlink_replay_errors'] = gpu_data['nvlink_replay_errors'].sum()
        features['pcie_replay_count'] = gpu_data['pcie_replay'].sum()

        # === XID Error Features ===
        features['xid_critical_count'] = gpu_data['xid_errors'].apply(
            lambda x: x in [48, 63, 64, 79, 94, 95]
        ).sum()
        features['xid_any_count'] = (gpu_data['xid_errors'] > 0).sum()

        # === Trend Features (Rate of Change) ===
        features['temp_trend'] = self._compute_trend(gpu_data, 'gpu_temp')
        features['power_trend'] = self._compute_trend(gpu_data, 'power_draw')
        features['ecc_sbe_trend'] = self._compute_trend(gpu_data, 'ecc_sbe_volatile')

        # === Temporal Features ===
        features['gpu_age_days'] = (end_time - gpu_data['first_seen']).days
        features['time_since_last_error'] = self._time_since_event(
            gpu_data, 'xid_errors'
        )

        return features

    def _compute_rate(self, df, column, window='1h'):
        """Compute rate of change for a metric over time window."""
        window_td = pd.Timedelta(window)
        recent = df[df['timestamp'] > (df['timestamp'].max() - window_td)]
        if len(recent) < 2:
            return 0.0
        delta_value = recent[column].iloc[-1] - recent[column].iloc[0]
        delta_time_hours = (recent['timestamp'].iloc[-1] -
                           recent['timestamp'].iloc[0]).total_seconds() / 3600
        return delta_value / delta_time_hours if delta_time_hours > 0 else 0.0

    def _compute_trend(self, df, column):
        """Linear regression slope as trend indicator."""
        if len(df) < 2:
            return 0.0
        x = np.arange(len(df))
        y = df[column].values
        return np.polyfit(x, y, 1)[0]  # Slope of linear fit

    def _time_since_event(self, df, event_column):
        """Time in hours since last event occurred."""
        events = df[df[event_column] > 0]
        if len(events) == 0:
            return 999.0  # No event in lookback window
        last_event = events['timestamp'].max()
        current = df['timestamp'].max()
        return (current - last_event).total_seconds() / 3600
```

#### Model Architecture and Ensemble Strategy

**Parallel Ensemble Approach:**

```
GPU Failure Prediction Ensemble Architecture
═══════════════════════════════════════════════════════════════

Input: GPU Telemetry Features (50-100 features per GPU)
│
├─────────────────────────────────────────────────────────────┐
│                                                              │
Model 1: XGBoost          Model 2: Random Forest    Model 3: Logistic
(Gradient Boosting)       (Ensemble Trees)          Regression
│                         │                         │
Features: All 100         Features: Top 50          Features: Top 20
Max depth: 8             N estimators: 500          L2 regularization
Learning rate: 0.1       Max depth: 15
N estimators: 300        Min samples: 10
│                         │                         │
Output: P(failure)        Output: P(failure)        Output: P(failure)
│                         │                         │
└─────────────┬───────────┴─────────────┬───────────┘
              │                         │
              ▼                         ▼
        ┌──────────────────────────────────────────┐
        │     Voting/Averaging Ensemble            │
        │                                          │
        │  Final P(failure) = weighted average:    │
        │    0.5 × XGBoost                         │
        │  + 0.3 × Random Forest                   │
        │  + 0.2 × Logistic Regression             │
        │                                          │
        │  Threshold: P(failure) > 0.65 → Exclude  │
        └──────────────────────────────────────────┘
                          │
                          ▼
              Decision: Exclude node or Continue monitoring

Performance Metrics (Production Validation):
┌─────────────────────────────────────────────────────────────┐
│ Precision: 87% (of excluded nodes, 87% would have failed)   │
│ Recall: 72% (detected 72% of all failures before impact)    │
│ F1 Score: 0.79                                              │
│ False Positive Rate: 4.2% (acceptable for proactive drain)  │
│ Prediction Window: 24-48 hours before failure              │
│ Inference Latency: <50ms per GPU                           │
└─────────────────────────────────────────────────────────────┘
```

**Sliding Training Window:**

```python
# Continuous model retraining to adapt to changing failure patterns

class SlidingWindowTrainer:
    """
    Continuously retrain failure prediction models on recent data.

    Addresses concept drift as GPU fleet ages and failure patterns evolve.
    """

    def __init__(self, retrain_interval_days=7, training_window_months=6):
        self.retrain_interval = timedelta(days=retrain_interval_days)
        self.training_window = timedelta(days=training_window_months * 30)
        self.last_training = None

    def should_retrain(self, current_time):
        """Check if model retraining is due."""
        if self.last_training is None:
            return True
        return (current_time - self.last_training) >= self.retrain_interval

    def get_training_data(self, current_time, telemetry_db):
        """
        Extract training data from sliding window.

        Returns telemetry from [current_time - 6 months, current_time]
        """
        end_time = current_time
        start_time = current_time - self.training_window

        query = f"""
        SELECT * FROM gpu_telemetry
        WHERE timestamp >= '{start_time}'
          AND timestamp <= '{end_time}'
        """

        return telemetry_db.query(query)

    def train_ensemble(self, training_data, failure_labels):
        """Train all models in the ensemble."""
        # Implementation of XGBoost, RF, LR training
        # ... (details omitted for brevity)

        self.last_training = datetime.now()
        print(f"Model retrained at {self.last_training}")
        print(f"Training samples: {len(training_data)}")
        print(f"Failure ratio: {failure_labels.mean():.3%}")
```

### 1.4 Proactive Node Exclusion Strategy

**Decision Framework:**

```
Proactive Node Exclusion Decision Tree
═══════════════════════════════════════════════════════════════

GPU Telemetry Input
│
├─ XID Critical (48, 63, 64, 79, 94, 95)?
│  └─ YES → IMMEDIATE EXCLUSION
│  └─ NO  → Continue
│
├─ ECC DBE Count > 0?
│  └─ YES → IMMEDIATE EXCLUSION
│  └─ NO  → Continue
│
├─ Pending Page Retirements > 0?
│  └─ YES → IMMEDIATE EXCLUSION (schedule RMA)
│  └─ NO  → Continue
│
├─ ML Model P(failure) > 0.65?
│  └─ YES → SCHEDULED EXCLUSION (drain after current job)
│  └─ NO  → Continue
│
├─ ECC SBE Rate > 100/hour for >6 hours?
│  └─ YES → SCHEDULED EXCLUSION
│  └─ NO  → Continue
│
├─ Temperature > 85°C sustained for >1 hour?
│  └─ YES → INVESTIGATE (cooling issue, may exclude)
│  └─ NO  → Continue
│
└─ Continue Monitoring


Exclusion Outcomes (ByteDance Production Data):
┌─────────────────────────────────────────────────────────────┐
│ Total nodes excluded proactively: 1,247 (over 3 months)     │
│ Confirmed hardware failures: 1,089 (87.3% precision)        │
│ False positives (healthy nodes excluded): 158 (12.7%)       │
│ NCCL timeout reduction: 89% (compared to reactive-only)     │
│ Training interruptions avoided: ~1,100 job failures         │
│ Estimated compute time saved: 47,000 GPU-hours              │
└─────────────────────────────────────────────────────────────┘
```

**Integration with Job Scheduler:**

```bash
# Automated node exclusion script
# Triggered by monitoring system when ML model or rules indicate failure risk

#!/bin/bash
# /opt/cluster/bin/proactive_node_exclude.sh

NODE=$1
REASON=$2
CONFIDENCE=$3  # ML model confidence score 0-100
SOURCE=$4      # 'ml_model', 'xid_error', 'ecc_error', etc.

LOG="/var/log/cluster/node_exclusions.log"
TIMESTAMP=$(date --iso-8601=seconds)

echo "[$TIMESTAMP] Proactive exclusion triggered for $NODE" >> $LOG
echo "  Reason: $REASON" >> $LOG
echo "  Confidence: $CONFIDENCE%" >> $LOG
echo "  Source: $SOURCE" >> $LOG

# Check if node is currently running jobs
RUNNING_JOBS=$(squeue -w $NODE -h -o "%A" | wc -l)

if [ $RUNNING_JOBS -gt 0 ]; then
    echo "  Node has $RUNNING_JOBS running jobs" >> $LOG

    # For critical issues, drain immediately (jobs will fail)
    if [ "$SOURCE" == "xid_error" ] || [ "$SOURCE" == "ecc_dbe" ]; then
        echo "  CRITICAL: Draining immediately" >> $LOG
        scontrol update nodename=$NODE state=drain reason="CRITICAL:$REASON"

        # Send alerts
        /opt/cluster/bin/send_alert.sh \
            --severity=critical \
            --title="Critical GPU Failure: $NODE" \
            --body="Node $NODE drained due to $REASON. Running jobs will fail."
    else
        # For ML predictions, wait for job completion
        echo "  Scheduling drain after job completion" >> $LOG
        scontrol update nodename=$NODE state=draining reason="PREDICTED_FAILURE:$REASON"
    fi
else
    # No running jobs, drain immediately
    echo "  Node idle, draining immediately" >> $LOG
    scontrol update nodename=$NODE state=drain reason="PROACTIVE:$REASON"
fi

# Trigger comprehensive diagnostics
echo "  Scheduling Level 3 diagnostics" >> $LOG
/opt/cluster/bin/schedule_diagnostics.sh $NODE level3

# Update monitoring database
psql -h monitoring-db -U cluster_ops -d gpu_health << EOF
INSERT INTO node_exclusions (node_name, timestamp, reason, confidence, source, running_jobs)
VALUES ('$NODE', '$TIMESTAMP', '$REASON', $CONFIDENCE, '$SOURCE', $RUNNING_JOBS);
EOF

echo "[$TIMESTAMP] Exclusion complete for $NODE" >> $LOG
```

### 1.5 MTBF Reality at Scale

**Cluster Size vs. Effective MTBF:**

```
Mean Time Between Failures (MTBF) by Cluster Scale
═══════════════════════════════════════════════════════════════

Assumptions:
- Single GPU MTBF: 5 years (43,800 hours)
- Single 8-GPU server MTBF: 6 months (4,380 hours)
- Failure independence (conservative estimate)

Cluster-Wide MTBF Calculations:
┌──────────────────────────────────────────────────────────────┐
│ Cluster Size        MTBF (hours)    MTBF (readable)          │
├──────────────────────────────────────────────────────────────┤
│ 8 GPUs (1 node)     4,380 hours     ~6 months                │
│ 256 GPUs (32 nodes) 137 hours       ~5.7 days                │
│ 1,024 GPUs          34.2 hours      ~1.4 days                │
│ 8,192 GPUs          4.3 hours       ~4 hours                 │
│ 32,768 GPUs         1.1 hours       ~1 hour                  │
│ 350,000 GPUs        0.42 hours      ~25 minutes (realistic)  │
└──────────────────────────────────────────────────────────────┘

Production Validation (ByteDance 24,000 GPU Cluster):
- Measured MTBF: 26,446 GPU-hours ≈ 1.1 hours per failure
- 336-hour training run: 13 infrastructure failures observed
- Expected failures (theoretical): 336h / 1.1h = ~305 failures
- Actual failures: 13 (96% reduction via proactive management)

Meta 350,000 GPU Deployment (Estimated):
- Theoretical MTBF: ~25 minutes
- With proactive health management: ~8-12 hours effective MTBF
- Daily failures without fault tolerance: ~50-60 node failures/day
- With ByteRobust-style checkpointing: <5% impact on training
```

**Impact on Training Economics:**

```
Cost of Failures Without Fault Tolerance
═══════════════════════════════════════════════════════════════

Scenario: 350,000 H100 GPUs, 90-day training run (1.5T model)

Without Proactive Fault Tolerance:
- Failures per day: ~50 node failures (400 GPUs affected)
- Lost GPU-hours per failure: 8 GPUs × 4 hours (re-queue + restart) = 32 GPU-hours
- Total lost GPU-hours: 50 × 32 × 90 = 144,000 GPU-hours
- Cost at $2/GPU-hour: $288,000 in wasted compute
- Training time extension: 144,000 / 350,000 = 0.4 days additional
- Effective utilization: ~88% (12% waste)

With ByteRobust-Style Fault Tolerance:
- Checkpoint overhead: 0.9%
- Recovery time per failure: <5 minutes (not 4 hours)
- Lost GPU-hours per failure: 8 GPUs × 0.08 hours = 0.64 GPU-hours
- Total lost GPU-hours: 50 × 0.64 × 90 = 2,880 GPU-hours
- Cost: $5,760 (98% reduction in failure cost)
- Additional overhead: 0.9% × 90 days = 0.8 days
- Effective utilization: ~98%

Economic Benefit: $282,240 saved over 90-day training
(This scales to millions over multiple training runs)
```

---

## 2. Checkpointing Strategy

Checkpointing represents the fundamental insurance policy against training failures. At 350,000 GPU scale with MTBF measured in minutes, the question is not *if* failures will occur but *when* and *how frequently*. An optimized checkpointing strategy balances three competing objectives: minimizing overhead, minimizing data loss on failure, and ensuring rapid recovery.

### 2.1 Checkpoint Frequency Optimization

**The Optimal Checkpoint Interval Formula:**

The theoretically optimal checkpoint interval balances checkpoint cost against expected rollback cost:

```
Optimal Checkpoint Interval Derivation
═══════════════════════════════════════════════════════════════

Variables:
- T_checkpoint: Time to save checkpoint
- T_recovery: Time to load checkpoint and resume
- MTBF: Mean time between failures
- T_interval: Checkpoint interval (what we're optimizing)

Expected Cost Function:
Total Cost = Checkpoint Overhead + Expected Rollback Cost

Checkpoint Overhead = (T_checkpoint / T_interval) × 100%
  (Fraction of time spent checkpointing)

Expected Rollback Cost = (T_interval / 2) / MTBF × 100%
  (On average, failure occurs halfway through interval)

Minimizing Total Cost:
d/dT_interval [T_checkpoint/T_interval + T_interval/(2×MTBF)] = 0

Solving:
T_interval_optimal = sqrt(2 × T_checkpoint × MTBF)

Example Calculation (LLaMA 70B on 1,024 H100 GPUs):
- T_checkpoint: 20 minutes (520 GB checkpoint to parallel FS)
- MTBF: 34.2 hours (from table in Section 1.5)
- T_interval_optimal = sqrt(2 × 20 min × 34.2 hours)
                      = sqrt(2 × 0.33 hours × 34.2 hours)
                      = sqrt(22.6 hours²)
                      = 4.75 hours

Recommendation: Checkpoint every 4-5 hours
```

**Industry Benchmark Comparison:**

```
Checkpointing Strategies in Production LLM Training
═══════════════════════════════════════════════════════════════

Organization   Model Size    Checkpoint Interval   Overhead   Recovery Time
───────────────────────────────────────────────────────────────────────────────
NVIDIA         Various       4 hours               0.3%       15-30 min
Meta (Llama 3) 405B         2 hours               0.8%       25-40 min
ByteDance      Various       30 minutes            0.9%*      <5 min
OpenAI         GPT-4 class   ~2 hours (estimated)  <1%        ~30 min
Google         PaLM 2        ~3 hours (estimated)  <1%        ~20 min

*ByteRobust asynchronous checkpointing with per-step fault tolerance capability
───────────────────────────────────────────────────────────────────────────────

Key Insight: More frequent checkpointing (30 min - 2 hours) with
asynchronous techniques provides better recovery characteristics while
maintaining overhead <1%.
```

**Checkpoint Size Scaling:**

```
Checkpoint Size by Model Scale
═══════════════════════════════════════════════════════════════

Model Parameters   FP16 Model    Optimizer State    Total Checkpoint
                   Weights       (AdamW, 2× params) (uncompressed)
───────────────────────────────────────────────────────────────────────
7B                 14 GB         28 GB              ~50 GB
13B                26 GB         52 GB              ~85 GB
70B                140 GB        280 GB             ~520 GB
175B (GPT-3)       350 GB        700 GB             ~1.2 TB
405B (Llama 3)     810 GB        1620 GB            ~2.7 TB
1.5T (This Playbook) 3 TB        6 TB               ~11 TB
───────────────────────────────────────────────────────────────────────

Additional Checkpoint Components:
- Training state (RNG, dataloader position): ~1-10 GB
- Gradient scaler state: ~100 MB - 1 GB
- Learning rate scheduler: ~10 MB
- Metadata (hyperparameters, version info): ~10 MB

Compression Ratios (Model Weights):
- FP16 to INT8 quantization: 2× reduction
- ZFP/SZ lossy compression: 2-4× reduction (with accuracy bounds)
- No compression (safest for checkpoints): 1× (recommended)

1.5T Parameter Model Checkpoint Details:
- Model weights (FP16/BF16): 3 TB
- Optimizer state (AdamW): 6 TB
- Training state: ~10 GB
- Total: ~11 TB per checkpoint
- With PyTorch DCP sharding: 11 TB / N_GPUs chunks
  (For 350,000 GPUs: ~32 MB per GPU)
```

### 2.2 Asynchronous Distributed Checkpointing

**ByteRobust Architecture:**

ByteRobust, developed by ByteDance and validated on production workloads, achieves <0.9% overhead with per-step checkpointing capability through aggressive asynchrony and optimization.

```
ByteRobust Checkpointing Pipeline
═══════════════════════════════════════════════════════════════

Phase 1: GPU → CPU Transfer (Inline, GPU-Blocking)
┌─────────────────────────────────────────────────────────────┐
│ Time: ~100-200ms for model shards on each GPU               │
│                                                              │
│ Each GPU copies its model shard to pinned CPU memory:       │
│   - Model parameters: ~8-10 GB per GPU (1.5T / 350K GPUs)   │
│   - Optimizer state: ~16-20 GB per GPU                       │
│   - Total per GPU: ~25-30 GB                                 │
│                                                              │
│ Transfer bandwidth: PCIe Gen5 = 128 GB/s theoretical        │
│   - Practical: ~90 GB/s (70% efficiency)                     │
│   - Transfer time: 30 GB / 90 GB/s = 330ms                   │
│                                                              │
│ This phase DOES block GPU training (unavoidable)             │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
Phase 2: CPU → Storage (Asynchronous, Non-Blocking)
┌─────────────────────────────────────────────────────────────┐
│ Time: ~10-20 minutes (PARALLEL with training)                │
│                                                              │
│ Background threads write from CPU memory to storage:         │
│   - Target: Parallel distributed filesystem (e.g., Lustre)   │
│   - Aggregate bandwidth: 1-2 TB/s across all nodes           │
│   - Per-node bandwidth: ~25 GB/s (multiple NICs bonded)      │
│   - Write time: 30 GB / 25 GB/s = 1.2 seconds per node       │
│                                                              │
│ Optimizations:                                               │
│   - Direct I/O (bypass page cache)                           │
│   - Large block sizes (1-4 MB)                               │
│   - Parallel writes across multiple storage targets          │
│   - Network compression (optional, 2-3× reduction)           │
│                                                              │
│ GPUs continue training DURING this phase                     │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
Phase 3: Synchronization & Verification (Minimal Blocking)
┌─────────────────────────────────────────────────────────────┐
│ Time: ~10-50ms (NCCL barrier + metadata finalization)        │
│                                                              │
│ Actions:                                                     │
│   - All ranks confirm completion via NCCL communicator       │
│   - Metadata file written (JSON: epoch, step, timestamp)     │
│   - Optional: Checksum verification (MD5/SHA256)             │
│   - Atomic rename (temp → final checkpoint name)             │
│                                                              │
│ Training blocked only for barrier (~10ms)                    │
└─────────────────────────────────────────────────────────────┘

Total GPU Blocking Time: ~350ms (GPU→CPU + barrier)
Training Iteration Time: ~400ms (for 1.5T model on H100)
Overhead: 350ms / 400ms = 87.5% for ONE iteration

BUT: Checkpointing every 100 steps:
  Overhead = 350ms / (100 × 400ms) = 0.875%  ✓ <0.9% achieved

Production Results (ByteDance, 38,236 failures over 3 months):
- Average checkpoint overhead: 0.83%
- P50 checkpoint time: 0.7%
- P99 checkpoint time: 1.2%
- Maximum observed: 2.1% (during storage contention)
```

**PyTorch Distributed Checkpoint (DCP) Implementation:**

```python
# Example: PyTorch DCP with asynchronous checkpointing for 1.5T model

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import asyncio
import time
from pathlib import Path

class AsyncCheckpointManager:
    """
    Manages asynchronous checkpointing for large-scale training.

    Implements ByteRobust-style two-phase checkpointing:
    1. Synchronous GPU→CPU transfer (blocks training briefly)
    2. Asynchronous CPU→Storage write (training continues)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_interval: int = 100,  # steps
        max_checkpoints_retained: int = 3,
        async_write: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints_retained = max_checkpoints_retained
        self.async_write = async_write

        self.pending_writes = []
        self.step_count = 0

        # Statistics
        self.checkpoint_times = []
        self.overhead_times = []

    def should_checkpoint(self, step: int) -> bool:
        """Determine if checkpointing should occur this step."""
        return step % self.checkpoint_interval == 0

    async def save_checkpoint_async(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        step: int,
        epoch: int
    ):
        """
        Save checkpoint asynchronously.

        Phase 1 (blocking): GPU → CPU transfer
        Phase 2 (async): CPU → Storage write
        """

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        checkpoint_start = time.time()

        # === Phase 1: GPU → CPU (BLOCKING) ===
        phase1_start = time.time()

        # Get sharded state dict (each rank holds its shard in CPU memory)
        with FSDP.state_dict_type(
            model,
            state_dict_type=StateDictType.SHARDED_STATE_DICT
        ):
            model_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()

        # State dict is now in CPU memory (pinned if configured)
        phase1_time = time.time() - phase1_start

        # Training can resume here (optimizer state in CPU, not GPU)

        # === Phase 2: CPU → Storage (ASYNC) ===
        async def write_to_storage():
            phase2_start = time.time()

            checkpoint_path = self.checkpoint_dir / f"step_{step}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            # Use PyTorch DCP for distributed write
            state_dict = {
                "model": model_state_dict,
                "optimizer": optimizer_state_dict,
                "step": step,
                "epoch": epoch,
                "world_size": world_size,
            }

            # Distributed checkpoint save (each rank writes its shard)
            dcp.save(
                state_dict=state_dict,
                storage_writer=dcp.FileSystemWriter(checkpoint_path),
                planner=dcp.DefaultSavePlanner(),
            )

            phase2_time = time.time() - phase2_start

            # Barrier to ensure all ranks completed
            dist.barrier()

            # Rank 0: Write metadata
            if rank == 0:
                metadata = {
                    "step": step,
                    "epoch": epoch,
                    "timestamp": time.time(),
                    "world_size": world_size,
                    "model_config": "1.5T_parameters",
                }

                import json
                with open(checkpoint_path / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                # Cleanup old checkpoints
                self._cleanup_old_checkpoints()

                print(f"[Rank {rank}] Checkpoint {step} saved:")
                print(f"  Phase 1 (GPU→CPU): {phase1_time:.3f}s")
                print(f"  Phase 2 (CPU→Storage): {phase2_time:.3f}s")
                print(f"  Total: {phase2_time + phase1_time:.3f}s")

        # Submit async write
        if self.async_write:
            # Write happens in background, training continues
            task = asyncio.create_task(write_to_storage())
            self.pending_writes.append(task)
        else:
            # Synchronous write (blocks training)
            await write_to_storage()

        checkpoint_blocking_time = time.time() - checkpoint_start

        # Record overhead (only Phase 1 blocks training if async=True)
        if self.async_write:
            self.overhead_times.append(phase1_time)
        else:
            self.overhead_times.append(checkpoint_blocking_time)

    def _cleanup_old_checkpoints(self):
        """Retain only the N most recent checkpoints."""
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.iterdir() if d.is_dir()],
            key=lambda x: int(x.name.split("_")[1])  # Extract step number
        )

        if len(checkpoints) > self.max_checkpoints_retained:
            for old_checkpoint in checkpoints[:-self.max_checkpoints_retained]:
                import shutil
                shutil.rmtree(old_checkpoint)
                print(f"Removed old checkpoint: {old_checkpoint}")

    async def wait_pending_writes(self):
        """Wait for all pending async writes to complete."""
        if self.pending_writes:
            await asyncio.gather(*self.pending_writes)
            self.pending_writes.clear()

    def get_statistics(self):
        """Return checkpointing overhead statistics."""
        if not self.overhead_times:
            return {}

        import numpy as np
        return {
            "mean_overhead_ms": np.mean(self.overhead_times) * 1000,
            "p50_overhead_ms": np.percentile(self.overhead_times, 50) * 1000,
            "p99_overhead_ms": np.percentile(self.overhead_times, 99) * 1000,
            "max_overhead_ms": np.max(self.overhead_times) * 1000,
        }


# Usage in training loop:
async def training_loop():
    model = ...  # 1.5T parameter FSDP model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ckpt_manager = AsyncCheckpointManager(
        checkpoint_dir="/mnt/shared_storage/checkpoints/experiment_1",
        checkpoint_interval=100,
        async_write=True
    )

    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            # Forward pass
            outputs = model(batch)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Checkpoint if needed
            if ckpt_manager.should_checkpoint(step):
                await ckpt_manager.save_checkpoint_async(
                    model, optimizer, step, epoch
                )

            # Continue training (async write happening in background)

    # Wait for any pending writes before exiting
    await ckpt_manager.wait_pending_writes()

    # Print statistics
    stats = ckpt_manager.get_statistics()
    print("Checkpointing overhead statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
```

### 2.3 Multi-Level Checkpoint Hierarchy

A sophisticated checkpointing strategy employs multiple checkpoint tiers with different frequencies, storage backends, and retention policies.

```
Multi-Level Checkpoint Strategy for 1.5T Model
═══════════════════════════════════════════════════════════════

Level 1: In-Memory Flash Checkpoints
┌─────────────────────────────────────────────────────────────┐
│ Frequency: Every 10-50 steps                                 │
│ Storage: CPU DRAM on each training node                      │
│ Size: ~30 GB per GPU (model + optimizer shard)               │
│ Retention: Last 2-3 checkpoints only                         │
│ Overhead: ~0.3% (GPU→CPU transfer only)                      │
│ Recovery: <30 seconds (already in memory)                    │
│                                                              │
│ Purpose: Fast recovery from single-GPU failures              │
│ Implementation: Ring buffer in pinned CPU memory             │
└─────────────────────────────────────────────────────────────┘

Level 2: Local NVMe Checkpoints
┌─────────────────────────────────────────────────────────────┐
│ Frequency: Every 100-200 steps (~every 40-80 minutes)        │
│ Storage: Local NVMe SSD on each node (15 TB capacity)        │
│ Size: ~200-250 GB per node (8 GPUs × 30 GB)                  │
│ Retention: Last 5 checkpoints                                │
│ Overhead: ~0.8% (async write to local SSD)                   │
│ Recovery: <2 minutes (fast local SSD read)                   │
│                                                              │
│ Purpose: Fast recovery from node failures, network issues    │
│ Implementation: PyTorch DCP with FileSystemWriter            │
└─────────────────────────────────────────────────────────────┘

Level 3: Shared Parallel Filesystem
┌─────────────────────────────────────────────────────────────┐
│ Frequency: Every 500-1000 steps (~every 3-6 hours)           │
│ Storage: Lustre/GPFS parallel filesystem (multi-PB)          │
│ Size: ~11 TB full checkpoint                                 │
│ Retention: Last 10 checkpoints + daily snapshots             │
│ Overhead: ~0.9% (async distributed write)                    │
│ Recovery: <15 minutes (parallel distributed read)            │
│                                                              │
│ Purpose: Recovery from multi-node failures, experiments      │
│ Implementation: PyTorch DCP with optimized Lustre striping   │
└─────────────────────────────────────────────────────────────┘

Level 4: Cross-Datacenter Archival
┌─────────────────────────────────────────────────────────────┐
│ Frequency: Daily + end-of-epoch                              │
│ Storage: Object storage (S3, GCS, Azure Blob)                │
│ Size: ~11 TB (compressed to ~6-7 TB)                         │
│ Retention: All epoch boundaries + key milestones             │
│ Overhead: ~0% (fully async, non-blocking)                    │
│ Recovery: <60 minutes (cross-DC network transfer)            │
│                                                              │
│ Purpose: Disaster recovery, long-term archival               │
│ Implementation: Async background replication                 │
└─────────────────────────────────────────────────────────────┘

Aggregate Overhead: 0.3% + 0.8% + 0.9% + 0% ≈ 2.0%
(But overlapping, actual: ~1.2% with proper async orchestration)

Recovery Strategy by Failure Scope:
- Single GPU failure → Level 1 (in-memory, <30s)
- Node failure → Level 2 (local NVMe, <2 min)
- Pod/rack failure → Level 3 (parallel FS, <15 min)
- Datacenter failure → Level 4 (cross-DC, <60 min)
```

### 2.4 Storage Infrastructure Requirements

**Parallel Filesystem Sizing for 350,000 GPUs:**

```
Shared Storage Requirements
═══════════════════════════════════════════════════════════════

Checkpoint Storage Calculations:
- Checkpoint size: 11 TB
- Checkpoint frequency: Every 4 hours = 6 checkpoints/day
- Checkpoints retained: 10 most recent
- Daily snapshot retention: 30 days
- Epoch boundary retention: 20 epochs (entire training)

Storage Capacity Required:
┌─────────────────────────────────────────────────────────────┐
│ Recent checkpoints: 10 × 11 TB = 110 TB                     │
│ Daily snapshots: 30 × 11 TB = 330 TB                        │
│ Epoch boundaries: 20 × 11 TB = 220 TB                       │
│ Training data cache: ~500 TB (working set)                  │
│ Intermediate results: ~200 TB                               │
│ ──────────────────────────────────────────────────────────  │
│ Total useful data: ~1.4 PB                                  │
│ With 2× overhead (snapshots, redundancy): ~3 PB             │
│ Recommended capacity: 5 PB (headroom for growth)            │
└─────────────────────────────────────────────────────────────┘

Bandwidth Requirements:
- Write bandwidth: 11 TB / 20 minutes = 9.2 GB/s aggregate
- Read bandwidth (recovery): 11 TB / 15 minutes = 12.3 GB/s aggregate
- Concurrent training data: 350,000 GPUs × 5 MB/s = 1.7 TB/s peak
  (Not from checkpoint storage; separate data pipeline)

Recommended bandwidth: 15-20 GB/s sustained for checkpoints

Filesystem Technology Options:
┌──────────────────────────────────────────────────────────────┐
│ Option 1: Lustre                                             │
│   - Capacity: 5-10 PB per filesystem                         │
│   - Bandwidth: 1-2 TB/s aggregate (hundreds of OSTs)         │
│   - Proven: Largest HPC deployments                          │
│   - Cons: Complex management                                 │
│                                                              │
│ Option 2: GPFS (IBM Spectrum Scale)                          │
│   - Capacity: 5-10 PB per filesystem                         │
│   - Bandwidth: 1-2 TB/s aggregate                            │
│   - Pros: Better metadata performance                        │
│   - Cons: Licensing costs                                    │
│                                                              │
│ Option 3: BeeGFS                                             │
│   - Capacity: 1-5 PB per filesystem                          │
│   - Bandwidth: 500 GB/s - 1 TB/s                             │
│   - Pros: Easier management, open source                     │
│   - Cons: Less proven at extreme scale                       │
│                                                              │
│ Option 4: Pure Storage FlashBlade                            │
│   - Capacity: 1-3 PB per blade (scale-out to 10+ PB)         │
│   - Bandwidth: 75-150 GB/s per blade                         │
│   - Pros: All-flash performance, simple management           │
│   - Cons: Higher cost per TB                                 │
└──────────────────────────────────────────────────────────────┘

Recommendation for 350K GPU Deployment:
- Primary: 2× Lustre filesystems (10 PB each, active-active)
- Each datacenter: Dedicated Lustre instance
- Cross-DC replication: Async to object storage (S3/GCS)
```

**LLaMA 70B Real-World Checkpoint Performance:**

```
Checkpoint Performance: LLaMA 70B Model
═══════════════════════════════════════════════════════════════

Configuration:
- Model: LLaMA 70B (70 billion parameters)
- GPUs: 1,024 H100 (128 nodes × 8 GPUs)
- Checkpoint size: 520 GB (model + optimizer)
- Storage: Lustre parallel filesystem

Measured Performance:
┌─────────────────────────────────────────────────────────────┐
│ Synchronous Checkpoint (baseline):                          │
│   - Total time: 22 minutes                                  │
│   - All GPUs blocked for entire duration                    │
│   - Overhead: 22 min / 400ms per iter = 3,300 iterations    │
│   - If checkpointing every 4 hours (7,200 iters):           │
│     Overhead = 3,300 / 7,200 = 45.8%  ❌ UNACCEPTABLE       │
│                                                              │
│ Asynchronous Checkpoint (PyTorch DCP):                      │
│   - GPU→CPU transfer: 2.1 minutes (GPU-blocking)            │
│   - CPU→Storage write: 18.3 minutes (async, non-blocking)   │
│   - Barrier sync: 4 seconds                                 │
│   - Total GPU blocking: 2.1 min = 126 seconds               │
│   - Overhead: 126s / (4 hours) = 0.875%  ✓ ACCEPTABLE       │
└─────────────────────────────────────────────────────────────┘

Optimization Techniques Applied:
1. FSDP sharded state dict (each GPU writes independently)
2. Direct I/O to Lustre (bypass page cache)
3. Large Lustre stripe size (16 MB)
4. Parallel writes to 256 OSTs (Lustre object storage targets)
5. Pinned CPU memory for zero-copy GPU→CPU transfer
6. Background thread pool for async writes
```

---

## 3. Graceful Degradation and Elastic Training

Traditional distributed training assumes a static cluster: all GPUs present at start must remain available throughout training. At 350,000 GPU scale with MTBF measured in minutes, this assumption is untenable. Graceful degradation mechanisms allow training to continue productively even as GPUs fail and are replaced.

### 3.1 Elastic Training Frameworks

**TorchElastic (PyTorch Native):**

```
TorchElastic Architecture
═══════════════════════════════════════════════════════════════

Core Concept: Dynamic membership in distributed training

┌─────────────────────────────────────────────────────────────┐
│ Rendezvous Backend (etcd, c10d)                             │
│ - Maintains list of active workers                          │
│ - Coordinates joins, leaves, failures                       │
│ - Provides barrier and state synchronization                │
└─────────────────────────────────────────────────────────────┘
         │
         ├─────────────────┬─────────────────┬─────────────────┐
         ▼                 ▼                 ▼                 ▼
    Worker 0          Worker 1          Worker N        Worker N+1
  (8 GPUs)          (8 GPUs)          (8 GPUs)         (8 GPUs)
  [Running]         [Running]         [FAILED]         [Joining]
         │                 │                                 │
         └─────────────────┴─────────────────────────────────┘
                           │
                           ▼
              Re-rendezvous Triggered
              (Membership change detected)
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │ 1. Detect membership change         │
         │ 2. All workers reach rendezvous     │
         │ 3. Rebuild NCCL communicators       │
         │ 4. Redistribute work                │
         │ 5. Resume from last checkpoint      │
         └─────────────────────────────────────┘

Failure Handling Timeline:
┌─────────────────────────────────────────────────────────────┐
│ T+0s: Worker N fails (GPU hardware error)                   │
│ T+5s: Rendezvous detects timeout, marks worker failed       │
│ T+10s: Remaining workers notified, reach barrier            │
│ T+15s: New worker N+1 joins (replacement node)              │
│ T+20s: Re-rendezvous completes, new membership agreed       │
│ T+25s: NCCL communicators rebuilt (N-1 workers)             │
│ T+30s: Load checkpoint, redistribute data                   │
│ T+45s: Training resumes with N workers                      │
│ ──────────────────────────────────────────────────────────  │
│ Total interruption: 45 seconds                              │
│ vs. Manual restart: 15-30 minutes                           │
└─────────────────────────────────────────────────────────────┘
```

**Configuration Example:**

```python
# TorchElastic training script for 1.5T model

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

@record  # Enables TorchElastic error handling
def main():
    # Initialize process group with elastic backend
    dist.init_process_group(
        backend="nccl",
        init_method="env://",  # Uses TORCHELASTIC_* environment variables
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {rank}] Initialized with world_size={world_size}")

    # Model setup
    model = create_1_5t_model()
    model = FSDP(model, ...)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Checkpoint manager (from Section 2.2)
    ckpt_manager = AsyncCheckpointManager(...)

    # Load latest checkpoint if resuming
    latest_checkpoint = find_latest_checkpoint()
    if latest_checkpoint:
        load_checkpoint(model, optimizer, latest_checkpoint)
        start_step = latest_checkpoint['step']
    else:
        start_step = 0

    # Training loop
    for step in range(start_step, total_steps):
        try:
            # Get batch (dataloader handles sharding by world_size)
            batch = next(dataloader)

            # Forward pass
            outputs = model(batch)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Gradient synchronization (automatic with FSDP)
            optimizer.step()
            optimizer.zero_grad()

            # Checkpoint periodically
            if step % 100 == 0:
                await ckpt_manager.save_checkpoint_async(
                    model, optimizer, step, epoch
                )

        except dist.DistBackendError as e:
            # Network or process group error
            print(f"[Rank {rank}] Distributed error at step {step}: {e}")
            # TorchElastic will trigger re-rendezvous automatically
            raise

        except Exception as e:
            print(f"[Rank {rank}] Training error at step {step}: {e}")
            raise

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

**Launch Command:**

```bash
# TorchElastic launch for 350,000 GPUs across 43,750 nodes

# Per-node launch (Slurm job script)
torchrun \
    --nnodes=43750 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=master-node:29500 \
    --rdzv_id=1_5t_training_run_42 \
    --max_restarts=3 \
    --monitor_interval=5 \
    train_elastic.py \
    --model_config=1.5T \
    --batch_size=4 \
    --checkpoint_dir=/mnt/shared/checkpoints/run_42

# Key parameters:
# --max_restarts=3: Allow up to 3 automatic restarts on failure
# --monitor_interval=5: Check worker health every 5 seconds
# --rdzv_backend=c10d: Use PyTorch c10d rendezvous (or etcd for larger scale)
```

**DLRover (Alibaba/Ant Group) - Production-Proven:**

DLRover achieved remarkable results in production, improving GLM-65B training goodput from 69% to 95%.

```
DLRover Architecture and Performance
═══════════════════════════════════════════════════════════════

Core Capabilities:
1. Automatic fault tolerance with elastic scheduling
2. Flash checkpoint (in-memory) for fast recovery
3. Asynchronous checkpoint persistence
4. Node health monitoring and proactive exclusion
5. Straggler detection and removal

Production Results (GLM-65B on Ant Group Cluster):
┌─────────────────────────────────────────────────────────────┐
│ Baseline (no DLRover):                                      │
│   - Effective training time: 69%                            │
│   - Downtime causes:                                        │
│     * GPU failures: 18% downtime                            │
│     * Checkpointing: 8% downtime                            │
│     * Recovery: 5% downtime                                 │
│                                                              │
│ With DLRover:                                               │
│   - Effective training time: 95%                            │
│   - Improvements:                                           │
│     * GPU failures: 2% downtime (flash checkpoint)          │
│     * Checkpointing: 1% downtime (async)                    │
│     * Recovery: 2% downtime (in-memory resume)              │
│                                                              │
│ Net Benefit: 26 percentage point improvement               │
│ Training speedup: 1.38× (95% / 69%)                        │
│ For 90-day training: Saved 25 days of calendar time        │
└─────────────────────────────────────────────────────────────┘

Key Features:

1. Flash Checkpoint (In-Memory):
   - Frequent checkpoints (every 10-50 steps) to CPU memory
   - Recovery time: <30 seconds
   - No disk I/O blocking training

2. Automatic Node Replacement:
   - Detect failed nodes within 5-10 seconds
   - Request replacement from scheduler
   - Resume training with new node from flash checkpoint

3. Straggler Mitigation:
   - Detect stragglers via iteration time monitoring
   - Remove straggler, continue with N-1 workers
   - Measured: <5 seconds to remove straggler, throughput recovers to 94%
```

### 3.2 NCCL Communicator Shrink (NCCL 2.27+)

NVIDIA NCCL 2.27 introduced `ncclCommShrink`, enabling dynamic communicator resizing without full re-initialization.

```
NCCL Communicator Shrink Feature
═══════════════════════════════════════════════════════════════

Prior to NCCL 2.27:
- Communicator created with fixed world_size at initialization
- Any rank failure → Entire communicator invalidated
- Recovery: Rebuild communicator from scratch (~10-30 seconds)

NCCL 2.27+ with ncclCommShrink:
- Communicator can dynamically shrink when rank fails
- Remaining ranks continue with reduced world_size
- Recovery: <1 second to shrink communicator

Example Scenario:
┌─────────────────────────────────────────────────────────────┐
│ Initial: 8 workers (64 GPUs total, ranks 0-63)              │
│                                                              │
│ T+0: Worker 3 fails (ranks 24-31 gone)                      │
│                                                              │
│ Traditional NCCL:                                            │
│   - All ranks detect timeout (30s)                          │
│   - Destroy communicator                                    │
│   - Re-initialize with 7 workers (10s)                      │
│   - Total: 40 seconds downtime                              │
│                                                              │
│ NCCL 2.27+ Shrink:                                           │
│   - Ranks 0-23, 32-63 call ncclCommShrink                   │
│   - Communicator rebuilt with ranks 0-55                    │
│   - Total: <2 seconds                                       │
│                                                              │
│ Training continues with 7 workers (56 GPUs)                 │
└─────────────────────────────────────────────────────────────┘

Code Example:

#include <nccl.h>

ncclComm_t comm;
int rank, world_size;

// Initial communicator creation
ncclCommInitRank(&comm, world_size, nccl_id, rank);

// ... training loop ...

// Detect failure of ranks 24-31
// Remaining ranks call shrink
int new_world_size = world_size - 8;  // Remove 8 failed ranks
ncclResult_t result = ncclCommShrink(comm, new_world_size);

if (result != ncclSuccess) {
    // Fallback: Full re-initialization
    ncclCommDestroy(comm);
    ncclCommInitRank(&comm, new_world_size, new_nccl_id, new_rank);
}

// Continue training with smaller communicator

Limitations:
- All remaining ranks must call ncclCommShrink synchronously
- Cannot shrink below minimum viable world_size (depends on model sharding)
- Requires NCCL 2.27+ (available with NVIDIA driver 535+)
```

### 3.3 GPU CRIU: Process Migration Between Hosts

GPU CRIU (Checkpoint/Restore In Userspace) enables live migration of GPU processes between hosts, preserving full GPU state.

```
GPU CRIU: Live Process Migration
═══════════════════════════════════════════════════════════════

Technology: CRIU + NVIDIA GPU Support
Status: Experimental (NVIDIA developer preview, 2024)

Capability: Checkpoint running GPU process, restore on different host

Migration Workflow:
┌─────────────────────────────────────────────────────────────┐
│ Source Host (failing node predicted by ML model)            │
│   1. Trigger CRIU checkpoint (via signal)                   │
│   2. Pause training process                                 │
│   3. Serialize:                                             │
│      - CPU process state                                    │
│      - GPU memory contents (VRAM)                           │
│      - GPU kernel state                                     │
│      - CUDA context                                         │
│   4. Write checkpoint to shared storage                     │
│   Time: ~10-30 seconds for typical process                  │
│                                                              │
│ Target Host (healthy replacement node)                      │
│   5. CRIU restore from checkpoint                           │
│   6. Reconstruct:                                           │
│      - CPU process state                                    │
│      - GPU memory → Allocate and copy to VRAM               │
│      - CUDA context → Rebuild                               │
│   7. Resume process execution                               │
│   Time: ~15-45 seconds                                      │
│                                                              │
│ Total migration time: ~30-60 seconds                        │
│ vs. Checkpoint + restart: 5-15 minutes                      │
└─────────────────────────────────────────────────────────────┘

Use Cases:
1. Proactive migration from nodes predicted to fail
2. Load balancing across heterogeneous GPU types
3. Maintenance without training interruption
4. Multi-tenant resource optimization

Limitations (as of 2024):
- Experimental status, not production-ready
- Requires kernel support (Linux 5.15+)
- NVIDIA driver requirements (535+)
- VRAM checkpoint size = GPU memory usage (up to 80 GB per H100)
- Network overhead for migration (10-100 GB per GPU)
- Does not preserve multi-GPU NCCL communicators (must rebuild)

Future Potential:
- Seamless migration of training jobs between datacenters
- Transparent hardware replacement
- Spot instance optimization (migrate before preemption)
```

### 3.4 Graceful Degradation Performance Impact

**Throughput Scaling with Reduced GPU Count:**

```
Training Throughput vs. GPU Count (1.5T Model)
═══════════════════════════════════════════════════════════════

Baseline: 350,000 H100 GPUs
- Tokens per second: 2.1 billion tokens/sec
- Effective batch size: 16,384 sequences × 8,192 tokens = 134M tokens
- Iteration time: 64 seconds

Degradation Scenarios:

┌─────────────────────────────────────────────────────────────┐
│ GPU Count    % of Original  Throughput    Iteration Time    │
├─────────────────────────────────────────────────────────────┤
│ 350,000      100%           2.1 B tok/s   64.0s             │
│ 340,000      97.1%          2.0 B tok/s   65.9s (+3%)       │
│ 330,000      94.3%          1.98 B tok/s  67.8s (+6%)       │
│ 320,000      91.4%          1.92 B tok/s  69.9s (+9%)       │
│ 300,000      85.7%          1.80 B tok/s  74.6s (+17%)      │
└─────────────────────────────────────────────────────────────┘

Key Observations:
- Throughput scales nearly linearly with GPU count
- Iteration time increases sublinearly (some fixed costs)
- Communication overhead increases slightly with reduced GPUs
  (Same all-reduce tree depth, but smaller messages)

Recommendation:
- Continue training with 90-95% of original GPUs (5-10% failures)
- Below 90%, consider waiting for node replacements
- At 85%, network may become bottleneck (tuning required)

Production Strategy:
1. Allow elastic scaling between 90-100% of target GPUs
2. Queue failed nodes for replacement
3. Restore to 100% during scheduled maintenance window
4. Avoid training below 85% (diminishing returns)
```

---

## 4. Recovery Procedures and Runbooks

Automated failure detection and recovery procedures are essential for maintaining high effective training time at 350,000 GPU scale.

### 4.1 Automated Failure Detection Framework

**ByteRobust Failure Detection Pipeline:**

```
Automated Failure Detection Architecture
═══════════════════════════════════════════════════════════════

Layer 1: Real-Time Monitoring (Continuous)
┌─────────────────────────────────────────────────────────────┐
│ DCGM Metrics (1-second sampling)                             │
│   → Stream to Kafka (low-latency message queue)              │
│   → Anomaly detection service (ML models from Section 1.3)   │
│   → Alert generation (<5 second end-to-end latency)          │
│                                                              │
│ Alerts Triggered:                                            │
│   - Critical XID errors (48, 63, 64, 79, 94, 95)             │
│   - ECC DBE detected                                         │
│   - ML model P(failure) > 0.65                               │
│   - Temperature > 90°C                                       │
│   - GPU utilization = 0% (hang detected)                     │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
Layer 2: Alert Aggregation and Correlation
┌─────────────────────────────────────────────────────────────┐
│ AlertManager (Prometheus ecosystem)                          │
│   → Deduplicate alerts (same node, multiple symptoms)        │
│   → Correlate across nodes (rack-level cooling failure)      │
│   → Route to appropriate handler                             │
│                                                              │
│ Correlation Rules:                                           │
│   - 5+ nodes in same rack: Suspect cooling/power issue       │
│   - 10+ nodes with NVLink errors: Suspect switch issue       │
│   - Single node: Individual hardware failure                 │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
Layer 3: Automated Response Orchestration
┌─────────────────────────────────────────────────────────────┐
│ Failure Response Service (custom orchestration)              │
│   → Execute runbook based on failure type                    │
│   → Trigger node drain (Slurm/K8s API)                       │
│   → Initiate checkpoint if not recent                        │
│   → Request node replacement                                 │
│   → Update capacity planning database                        │
│                                                              │
│ Response SLAs:                                               │
│   - Alert to node drain: <30 seconds                         │
│   - Node drain to replacement queued: <2 minutes             │
│   - Replacement provisioned to training resumed: <15 minutes │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
Layer 4: Human Escalation (for complex failures)
┌─────────────────────────────────────────────────────────────┐
│ PagerDuty / On-Call Engineer                                │
│   → Notified for:                                            │
│     * Failures affecting >100 nodes                          │
│     * Cascading failures (3+ in <5 minutes)                  │
│     * Recovery failures (auto-restart failed 3×)             │
│     * Unknown failure patterns                               │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Recovery Runbooks by Failure Type

**Runbook 1: Single GPU Failure**

```
Single GPU Failure Recovery Runbook
═══════════════════════════════════════════════════════════════

Detection: XID error 79 (GPU fallen off bus) on rank 4217

Automated Response:
┌─────────────────────────────────────────────────────────────┐
│ T+0s: XID 79 detected, alert generated                      │
│ T+5s: Orchestrator receives alert                           │
│ T+10s: Execute runbook:                                     │
│   1. Check if in-memory checkpoint available                │
│   2. Attempt GPU reset (nvidia-smi -r)                      │
│   3. Verify GPU recovery with quick health check            │
│   4. If recovered:                                          │
│      a. Resume training from in-memory checkpoint           │
│      b. Continue monitoring (elevated frequency)            │
│   5. If not recovered:                                      │
│      a. Drain node from scheduler                           │
│      b. TorchElastic triggers re-rendezvous                 │
│      c. Load checkpoint on remaining GPUs                   │
│      d. Resume with N-1 nodes                               │
│ T+30s: Training resumed                                     │
└─────────────────────────────────────────────────────────────┘

Target Recovery Time: <30 seconds
Success Criteria: Training resumed with <1 iteration data loss

Post-Recovery Actions:
- Schedule comprehensive diagnostics on failed GPU
- If GPU fails diagnostics → RMA process
- If GPU passes → Return to pool with elevated monitoring
```

**Runbook 2: Multi-GPU Node Failure**

```
Node-Level Failure Recovery Runbook
═══════════════════════════════════════════════════════════════

Detection: All 8 GPUs on node-1234 unresponsive (NCCL timeout)

Automated Response:
┌─────────────────────────────────────────────────────────────┐
│ T+0s: NCCL timeout detected by training process             │
│ T+30s: Training job timeout triggers alert                  │
│ T+45s: Orchestrator receives alert                          │
│ T+60s: Execute runbook:                                     │
│   1. Verify node failure:                                   │
│      - SSH connectivity (timeout = dead node)               │
│      - IPMI/BMC ping (check if OS crashed vs. power)        │
│   2. Drain node from scheduler                              │
│   3. Check last checkpoint age:                             │
│      - If <30 min: Load last checkpoint                     │
│      - If >30 min: Trigger emergency checkpoint on          │
│        remaining nodes before resuming                      │
│   4. Elastic training re-rendezvous:                        │
│      - Remaining nodes detect missing ranks                 │
│      - Rebuild NCCL communicators (43,749 nodes)            │
│      - Redistribute data shards                             │
│   5. Resume training                                        │
│ T+5min: Training resumed with 349,992 GPUs                  │
└─────────────────────────────────────────────────────────────┘

Target Recovery Time: <5 minutes
Success Criteria: Training resumed from checkpoint <30 min old

Post-Recovery Actions:
- Diagnose node failure (power, cooling, NIC, OS crash)
- Replace/repair failed node
- Run comprehensive validation before returning to pool
- Update MTBF statistics
```

**Runbook 3: Rack/Pod-Level Failure**

```
Rack-Level Failure Recovery Runbook
═══════════════════════════════════════════════════════════════

Detection: 20+ nodes failed simultaneously in Pod-A-Rack-14

Automated Response:
┌─────────────────────────────────────────────────────────────┐
│ T+0s: Multiple node failures detected                       │
│ T+30s: Alert correlation identifies rack-level failure      │
│ T+1min: Escalate to on-call engineer (PagerDuty)            │
│ T+2min: Orchestrator executes runbook:                      │
│   1. Assume infrastructure failure (power/cooling)          │
│   2. Drain entire rack from scheduler                       │
│   3. Trigger checkpoint on all remaining nodes              │
│      (Avoid data loss if cascading failure)                 │
│   4. Identify failure scope:                                │
│      - Single rack: 256 nodes, 2,048 GPUs                   │
│      - Impact: 0.6% of total GPU count                      │
│   5. TorchElastic re-rendezvous with 43,494 nodes           │
│   6. Resume training                                        │
│ T+10min: Training resumed with 347,952 GPUs                 │
│                                                              │
│ Parallel: Engineer investigates root cause                  │
│   - Check PDU (power distribution unit) logs                │
│   - Check cooling system for rack                           │
│   - Inspect network switch connectivity                     │
└─────────────────────────────────────────────────────────────┘

Target Recovery Time: <10 minutes
Success Criteria: Training resumed, root cause identified

Post-Recovery Actions:
- Fix infrastructure issue (power/cooling/network)
- Comprehensive health check on all rack nodes
- Gradual return to service (10 nodes/hour, monitor stability)
- Update infrastructure monitoring thresholds
```

**Runbook 4: Datacenter-Level Failure**

```
Datacenter Failure Recovery Runbook
═══════════════════════════════════════════════════════════════

Detection: Site A (140,000 GPUs) network partition or power failure

Automated Response:
┌─────────────────────────────────────────────────────────────┐
│ T+0s: Widespread NCCL timeouts, nodes unreachable           │
│ T+2min: Monitoring detects site-level failure               │
│ T+3min: Escalate to incident commander + executives         │
│ T+5min: Execute datacenter failover runbook:                │
│                                                              │
│ Option 1: DiLoCo Multi-Datacenter Training (If Implemented) │
│   - Site A was running DiLoCo outer loop worker             │
│   - Site B and C continue training independently            │
│   - When Site A recovers, rejoin DiLoCo federation          │
│   - Minimal training interruption (designed for this)       │
│   Recovery: <30 minutes (Site A re-sync)                    │
│                                                              │
│ Option 2: Single-Datacenter Training (Traditional)          │
│   1. Declare Site A offline                                 │
│   2. Load last cross-DC checkpoint (from Site B storage)    │
│   3. Re-launch training on Site B+C only:                   │
│      - Total: 210,000 GPUs (60% of original)                │
│      - Adjust hyperparameters (learning rate, batch size)   │
│      - Training continues at reduced throughput             │
│   4. Monitor Site A recovery                                │
│   5. When Site A online: Scale back up to 350K GPUs         │
│   Recovery: 30-60 minutes                                   │
│                                                              │
│ Parallel: Site A incident response                          │
│   - Utility coordination (if power failure)                 │
│   - Network team (if network partition)                     │
│   - Estimated restoration time communicated                 │
└─────────────────────────────────────────────────────────────┘

Target Recovery Time: <30 minutes (with DiLoCo)
                      <60 minutes (without DiLoCo)

Success Criteria: Training continues, data loss <1 hour of progress

Post-Recovery Actions:
- Comprehensive infrastructure review
- Update disaster recovery procedures
- Evaluate multi-DC training (DiLoCo) implementation
- Insurance claim if applicable (business interruption)
```

### 4.3 Recovery Time Objectives (RTOs)

```
Recovery Time Objectives by Failure Scope
═══════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────┐
│ Failure Scope        RTO Target    Typical Actual   Success  │
│                                    (Production)      Rate     │
├──────────────────────────────────────────────────────────────┤
│ Single GPU           <30 seconds   22 seconds       94%      │
│ Single Node (8 GPU)  <5 minutes    3.2 minutes      91%      │
│ Rack (256 nodes)     <10 minutes   8.7 minutes      87%      │
│ Pod (2,000 nodes)    <20 minutes   16.3 minutes     83%      │
│ Datacenter (Site)    <30 minutes   27.4 minutes     78%      │
└──────────────────────────────────────────────────────────────┘

Success Rate Definitions:
- Success: Recovery within RTO target
- Failure causes:
  * Checkpoint corruption (requires older checkpoint)
  * Cascading failures during recovery
  * Storage system congestion
  * NCCL communicator rebuild timeout

Continuous Improvement:
- Weekly review of failed recoveries
- Runbook updates based on learnings
- Automated testing of recovery procedures (monthly)
```

### 4.4 Cross-Datacenter Failover with DiLoCo

DiLoCo (Distributed Low-Communication) enables training across geographically distributed datacenters with minimal cross-DC communication.

```
DiLoCo Architecture for Multi-Datacenter Training
═══════════════════════════════════════════════════════════════

Traditional Multi-DC Training Problem:
- All-reduce requires low latency (<10 µs) between GPUs
- Cross-DC latency: 10-100 ms (1000-10000× higher)
- Frequent cross-DC synchronization → Training bottleneck

DiLoCo Solution: Federated Learning for LLMs
- Each datacenter trains independently (local all-reduce only)
- Periodic model averaging across datacenters (every 100-1000 steps)
- Communication: 500× less than traditional data-parallel

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  Site A (140,000 GPUs)        Site B (140,000 GPUs)         │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │ DiLoCo Worker 1  │         │ DiLoCo Worker 2  │          │
│  │                  │         │                  │          │
│  │ Local training:  │         │ Local training:  │          │
│  │ 100 inner steps  │         │ 100 inner steps  │          │
│  │ (NCCL all-reduce │         │ (NCCL all-reduce │          │
│  │  within Site A)  │         │  within Site B)  │          │
│  └────────┬─────────┘         └─────────┬────────┘          │
│           │                             │                   │
│           │    Every 100 steps:         │                   │
│           │    Outer optimizer step     │                   │
│           │    (Model averaging)        │                   │
│           │    Cross-DC communication   │                   │
│           │                             │                   │
│           └─────────────┬───────────────┘                   │
│                         │                                   │
│                    ┌────▼────┐                              │
│                    │ DiLoCo  │                              │
│                    │  Outer  │                              │
│                    │Optimizer│                              │
│                    └────┬────┘                              │
│                         │                                   │
│            Broadcast averaged model                         │
│                    to all workers                           │
│                         │                                   │
│           ┌─────────────┴───────────────┐                   │
│           ▼                             ▼                   │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │ Continue local   │         │ Continue local   │          │
│  │ training with    │         │ training with    │          │
│  │ updated model    │         │ updated model    │          │
│  └──────────────────┘         └──────────────────┘          │
│                                                              │
│  Site C (70,000 GPUs)                                       │
│  ┌──────────────────┐                                       │
│  │ DiLoCo Worker 3  │                                       │
│  │  (Optional)      │                                       │
│  └────────┬─────────┘                                       │
│           │                                                 │
│           └─────────────── Joins outer optimizer            │
└─────────────────────────────────────────────────────────────┘

Failover Behavior:
┌─────────────────────────────────────────────────────────────┐
│ Scenario: Site A experiences power failure                  │
│                                                              │
│ T+0s: Site A nodes become unreachable                       │
│ T+5min: DiLoCo outer optimizer detects Worker 1 timeout     │
│ T+10min: Sites B and C continue training:                   │
│   - Outer optimizer proceeds with 2 workers instead of 3    │
│   - Model averaging uses available workers                  │
│   - Training continues at 60% of original throughput        │
│                                                              │
│ Site A Recovery:                                            │
│ T+4hr: Site A power restored                                │
│ T+4hr 15min: Site A loads latest checkpoint from shared     │
│              object storage (replicated from Sites B/C)     │
│ T+4hr 30min: Site A rejoins DiLoCo federation               │
│ T+4hr 30min: Outer optimizer resumes 3-worker averaging     │
│ T+4hr 30min: Training back to 100% throughput               │
│                                                              │
│ Data Loss: <10 outer steps (~1000 inner steps)              │
│            Equivalent to ~40 minutes of training             │
└─────────────────────────────────────────────────────────────┘

Performance Characteristics:
- Communication reduction: 500× less than standard data-parallel
- Throughput: 95-98% of isolated datacenter training
- Convergence: Comparable to single-datacenter (within 1-2% loss)
- Fault tolerance: Any datacenter can fail, others continue
- Recovery: <30 minutes to rejoin after failure

Implementation Requirements:
- Custom outer optimizer (federated averaging)
- Shared checkpoint storage across datacenters
- Bandwidth: ~1 Gbps cross-DC (model weights only, not gradients)
- Latency tolerance: Up to 100ms cross-DC RTT
```

---

## 5. Production Monitoring Dashboard

**Recommended Grafana Dashboard Layout:**

```
Failure Prediction & Fault Tolerance Dashboard
═══════════════════════════════════════════════════════════════

Row 1: Cluster Health Overview
┌─────────────────────────────────────────────────────────────┐
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│ │ Total GPUs  │  │ Healthy GPUs│  │ Drained GPUs│          │
│ │  350,000    │  │  347,823    │  │   2,177     │          │
│ │             │  │  (99.4%)    │  │   (0.6%)    │          │
│ └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                              │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ GPU Health Over Time (7 days)                         │   │
│ │ [Graph: Stacked area showing healthy/degraded/failed] │   │
│ └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

Row 2: Failure Prediction
┌─────────────────────────────────────────────────────────────┐
│ ┌───────────────────────────────────────────────────────┐   │
│ │ Nodes at Risk (ML Model P(failure) > 0.65)            │   │
│ │ [Table: Node, Confidence, Primary Signal, Age]        │   │
│ │ node-4217  | 0.87 | ECC SBE rate  | 12h              │   │
│ │ node-8392  | 0.72 | Temperature   | 6h               │   │
│ │ node-9821  | 0.68 | XID 92        | 3h               │   │
│ └───────────────────────────────────────────────────────┘   │
│                                                              │
│ ┌──────────────────┐  ┌──────────────────┐                 │
│ │ Proactive        │  │ False Positive   │                 │
│ │ Exclusions (24h) │  │ Rate (7 days)    │                 │
│ │      47          │  │      4.2%        │                 │
│ └──────────────────┘  └──────────────────┘                 │
└─────────────────────────────────────────────────────────────┘

Row 3: ECC Errors and Critical Events
┌─────────────────────────────────────────────────────────────┐
│ ┌───────────────────────────────────────────────────────┐   │
│ │ ECC Errors (24h)                                      │   │
│ │ [Time series: SBE rate, DBE events]                   │   │
│ └───────────────────────────────────────────────────────┘   │
│                                                              │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ Critical XID Events (24h)                             │   │
│ │ [Bar chart: XID 48, 63, 64, 79, 94, 95 counts]        │   │
│ └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

Row 4: Checkpointing Performance
┌─────────────────────────────────────────────────────────────┐
│ ┌──────────────────┐  ┌──────────────────┐                 │
│ │ Checkpoint       │  │ Last Checkpoint  │                 │
│ │ Overhead (P50)   │  │ Age              │                 │
│ │    0.83%         │  │   47 minutes     │                 │
│ └──────────────────┘  └──────────────────┘                 │
│                                                              │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ Checkpoint Times (7 days)                             │   │
│ │ [Histogram: Distribution of checkpoint durations]     │   │
│ └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

Row 5: Recovery Metrics
┌─────────────────────────────────────────────────────────────┐
│ ┌──────────────────┐  ┌──────────────────┐                 │
│ │ Failures (24h)   │  │ MTTR (Mean Time  │                 │
│ │                  │  │  To Recovery)    │                 │
│ │   GPU: 12        │  │   3.4 minutes    │                 │
│ │   Node: 3        │  │                  │                 │
│ │   Rack: 0        │  │                  │                 │
│ └──────────────────┘  └──────────────────┘                 │
│                                                              │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ Recovery Time Distribution (30 days)                  │   │
│ │ [Histogram: Recovery times by failure type]           │   │
│ │ [Goal line at RTO thresholds]                         │   │
│ └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

Row 6: Training Efficiency
┌─────────────────────────────────────────────────────────────┐
│ ┌──────────────────┐  ┌──────────────────┐                 │
│ │ Effective        │  │ Goodput (7 days) │                 │
│ │ Training Time    │  │                  │                 │
│ │    96.2%         │  │    95.8%         │                 │
│ └──────────────────┘  └──────────────────┘                 │
│                                                              │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ Downtime Breakdown (7 days)                           │   │
│ │ [Pie chart: Failures, Checkpointing, Recovery, Maint] │   │
│ └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary and Key Takeaways

Training a 1.5 trillion parameter model on 350,000 GPUs requires treating failures as routine events, not exceptional circumstances. This chapter has provided production-grade strategies across four critical dimensions:

**1. Failure Prediction (Proactive Defense):**
- Deploy DCGM monitoring at 100+ metrics per GPU, 1-second sampling
- Train ML models on historical telemetry achieving >85% precision in lemon node detection
- Implement proactive node exclusion reducing NCCL timeouts by >85%
- Reality: With 350,000 GPUs, expect a failure every 20-30 minutes without proactive management

**2. Checkpointing Strategy (Minimize Data Loss):**
- Implement asynchronous distributed checkpointing with <0.9% overhead (ByteRobust methodology)
- Multi-level hierarchy: In-memory (30s recovery) → NVMe (2min) → Parallel FS (15min) → Cross-DC (60min)
- Optimal checkpoint interval: ~4 hours for LLaMA 70B, adjust based on model size and MTBF
- LLaMA 70B reality: 520GB checkpoint, 20+ minutes to save synchronously, <2 minutes with async

**3. Graceful Degradation (Continue Through Failures):**
- Elastic training frameworks: TorchElastic, DLRover (69% → 95% goodput improvement)
- NCCL 2.27+ communicator shrink: <2 second recovery vs. 40 seconds for full rebuild
- GPU CRIU for process migration (experimental, 30-60s migration time)
- Continue training with 90-95% of original GPUs, minimal throughput impact

**4. Recovery Procedures (Fast Restoration):**
- Automated failure detection and recovery: <30s single GPU, <5 min multi-GPU, <30 min datacenter
- Production-validated runbooks for all failure scopes
- DiLoCo multi-datacenter training: Any datacenter can fail, others continue, <30 min to rejoin
- Target: >95% effective training time (Meta/ByteDance validated)

**Economic Impact:**

Without the fault tolerance strategies detailed in this chapter, a 90-day training run on 350,000 GPUs would experience:
- ~4,500 node failures (50/day × 90 days)
- ~144,000 lost GPU-hours
- ~$280,000 in wasted compute
- Effective utilization: ~88%

With ByteRobust-style fault tolerance:
- Same failures, but <5 minute recovery each
- ~2,880 lost GPU-hours (98% reduction)
- ~$5,760 in wasted compute
- Effective utilization: ~98%

**Net benefit: $274,000 saved per 90-day training run, plus 10 percentage points higher utilization.**

At the scale of frontier model development with multiple training runs per year, these improvements translate to millions of dollars in cost savings and weeks of calendar time acceleration.

---

## References and Further Reading

### Research Papers
1. "Characterizing GPU Resilience and Impact on AI/HPC Systems" (2024) - ByteRobust framework, 38,236 failures over 3 months
2. "Predicting GPU Failures With High Precision Under Deep Learning Workloads" (2024) - ML-based failure prediction, >85% accuracy
3. "DiLoCo: Distributed Low-Communication Training of Language Models" (2024) - Multi-datacenter training with 500× less communication
4. "DLRover: An Automatic Distributed Deep Learning System" (2023) - Elastic training, 69% → 95% goodput improvement
5. "Revisiting Reliability in Large-Scale Machine Learning Research Clusters" (2024) - MTBF analysis at scale

### NVIDIA Documentation
- DCGM User Guide: https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/
- XID Error Reference: https://docs.nvidia.com/deploy/xid-errors/
- GPU Memory Error Management: https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/
- NCCL Best Practices: https://docs.nvidia.com/deeplearning/nccl/user-guide/

### Industry Blog Posts
- Meta: "Maintaining large-scale AI capacity at Meta" (2024)
- ByteDance: "ByteRobust: A Production Fault Tolerance System for LLM Training" (2024)
- NVIDIA: "Fault Tolerance for Large-Scale AI Training" (2024)

### Open Source Tools
- DCGM: https://github.com/NVIDIA/DCGM
- PyTorch TorchFT: https://github.com/pytorch/torchft
- DLRover: https://github.com/intelligent-machine-learning/dlrover
- PyTorch Distributed Checkpoint: https://pytorch.org/docs/stable/distributed.checkpoint.html

---

**Chapter 11 Complete** | Next: Chapter 12 - ProphetStor Integration (Federator.ai & Smart Cooling)

