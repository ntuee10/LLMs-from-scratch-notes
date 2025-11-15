# Chapter 2: Infrastructure Planning and Power Architecture

**Large-Scale LLM Training Playbook: 1.5 Trillion Parameter Model**
**Target Infrastructure: 5GW Multi-Datacenter Deployment**
**Investment Scale: $100+ Billion**

---

## Executive Overview

Power is the fundamental constraint that shapes every decision in trillion-parameter model training. While computational density, network bandwidth, and cooling capacity all impose limits, electrical infrastructure ultimately determines the feasible scale of any datacenter deployment. This chapter provides detailed specifications for planning, designing, and operating the electrical systems required to deliver 5 gigawatts of power across multiple datacenter sites.

**Key Insights from Production Deployments:**

- **Alibaba HPN**: 18MW building power constraint = ~15,000 GPU capacity per pod
- **Meta's 350,000 H100 deployment**: Requires ~462 MW facility power at PUE 1.20
- **Single-datacenter practical limit**: ~1.0-1.5 GW due to utility provisioning constraints
- **Multi-site necessity**: 5GW deployment requires 2-3 geographically distributed datacenters

**Chapter Roadmap:**

This chapter addresses five critical domains:

1. **Power Requirements and Sizing**: Calculating total power demand from GPU count through facility overhead
2. **Electrical Infrastructure Design**: Substations, transformers, UPS systems, and distribution architecture
3. **Multi-Site Power Distribution**: Allocating 5GW across Sites A (2.0 GW), B (2.0 GW), and C (1.0 GW)
4. **Power Optimization Strategies**: AI-driven power management, DVFS, and grid demand response
5. **Cost Analysis**: Capital expenditure ($15-20B) and operational expenses ($500M-750M annually)

**Success Criteria:**

- PUE (Power Usage Effectiveness) < 1.20 across all sites
- N+1 or 2N redundancy for critical electrical systems
- 99.99% uptime (52 minutes downtime annually)
- Grid-responsive power scheduling to optimize electricity costs

---

## 1. Power Requirements and Sizing

### 1.1 GPU Power Consumption: The Foundation

Modern accelerated computing imposes unprecedented power demands per compute unit. Understanding the full system power draw—not just GPU TDP—is essential for accurate capacity planning.

#### NVIDIA H100 System Power (8-GPU Server)

**Per-GPU Power Breakdown:**
```
Component                    Power Draw
─────────────────────────────────────────
GPU (H100 80GB SXM)         700W TDP
├─ Peak compute              700W
├─ Typical sustained         600-650W (85-93%)
└─ Idle                      50-75W

CPU (Dual-socket Xeon/EPYC) 200W
├─ 2× CPUs @ 100W each
└─ Data preprocessing, orchestration

System Memory (2TB DDR5)    50W
├─ 16× 128GB DIMMs
└─ High-bandwidth memory subsystem

Fans & Cooling              50W
├─ Chassis fans
└─ PSU active cooling

Network Interface (9 NICs)  50W
├─ 8× 400G NICs (Alibaba HPN config)
└─ ConnectX-7 adapters

PSU Losses (~10%)           100W
├─ Power supply inefficiency
└─ Conversion losses

─────────────────────────────────────────
TOTAL PER GPU (amortized)   ~1,150W
TOTAL PER 8-GPU SERVER      9,200W (9.2 kW)
```

**Validated Against Production:**
- Alibaba HPN: 18MW building / 15,000 GPUs = 1,200W per GPU (includes infrastructure)
- Our calculation: 1,150W per GPU (server only) + infrastructure overhead = 1,200W+ total

#### Phase 1 Deployment: 350,000 H100 GPUs

**IT Equipment Power Load:**
```
Component                          Quantity    Unit Power    Total Power
────────────────────────────────────────────────────────────────────────
H100 GPUs                          350,000     700W          245.0 MW
CPUs (dual-socket servers)         43,750      200W          8.8 MW
System Memory                      43,750      50W           2.2 MW
Fans & Cooling (server-level)      43,750      50W           2.2 MW
Network Interfaces                 43,750      50W           2.2 MW
PSU Losses                         43,750      100W          4.4 MW
────────────────────────────────────────────────────────────────────────
SUBTOTAL: IT Load                                            264.8 MW
Network Switches & Fabric          ~5,000      ~1.5 kW       7.5 MW
Storage Systems (NVMe, parallel FS) Variable                 10.0 MW
Management & Monitoring            Variable                  2.7 MW
────────────────────────────────────────────────────────────────────────
TOTAL IT EQUIPMENT POWER                                     285.0 MW
```

**Facility Power with PUE:**

Power Usage Effectiveness (PUE) accounts for all non-IT loads: cooling, power distribution losses, lighting, and facilities management.

```
PUE Scenario Analysis (350,000 GPUs, 285 MW IT Load)

PUE     Formula                    Facility Power    Annual Cost*
────────────────────────────────────────────────────────────────────
1.50    285 MW / 0.667 = 427.5 MW  427.5 MW         $1.50 billion
1.30    285 MW / 0.769 = 370.6 MW  370.6 MW         $1.30 billion
1.20    285 MW / 0.833 = 342.1 MW  342.1 MW         $1.20 billion
1.15    285 MW / 0.870 = 327.6 MW  327.6 MW         $1.15 billion

*Annual cost at $0.04/kWh, 8,760 hours/year, 100% utilization
```

**Key Insight**: Reducing PUE from 1.30 to 1.20 saves **$100M annually** in electricity costs at 350,000 GPU scale.

#### Phase 2 Expansion: 700,000-1,000,000 GPUs

The 5GW contracted capacity enables significant future expansion:

**700,000 GPU Scenario:**
- IT equipment power: 570 MW
- Facility power (PUE 1.20): 684 MW
- Contracted capacity per site: ~1.2-1.5 GW (with N+1 redundancy)

**1,000,000 GPU Scenario:**
- IT equipment power: 815 MW
- Facility power (PUE 1.20): 978 MW
- Contracted capacity per site: ~1.5-2.0 GW (with N+1 redundancy)

**5GW Total Allocation:**
- Site A: 2.0 GW contracted (supports 350K-400K GPUs)
- Site B: 2.0 GW contracted (supports 350K-400K GPUs)
- Site C: 1.0 GW contracted (supports 175K-200K GPUs)

This provides capacity for **900,000-1,000,000 GPUs** at Phase 2 buildout.

### 1.2 PUE Calculations and Targets

Power Usage Effectiveness is the industry-standard metric for datacenter energy efficiency:

**PUE = (Total Facility Power) / (IT Equipment Power)**

A PUE of 1.0 would represent perfect efficiency (all power to IT equipment, zero overhead). Real-world datacenters range from 1.15 (best-in-class) to 2.0+ (legacy facilities).

#### Industry Benchmarks

**Traditional Air-Cooled Datacenters:**
- PUE: 1.5-1.7 typical
- Cooling: CRAC/CRAH units, raised floor
- Examples: Most enterprise datacenters (pre-2020)

**Modern Air-Cooled (Hot Aisle Containment):**
- PUE: 1.3-1.4
- Cooling: Contained hot aisles, direct air cooling
- Examples: Recent hyperscale deployments

**Liquid-Cooled (Rear-Door Heat Exchangers):**
- PUE: 1.2-1.3
- Cooling: Rear-door coolers on racks, chilled water loops
- Examples: Meta, Google modern datacenters

**Liquid-Cooled (Direct-to-Chip):**
- PUE: 1.15-1.25
- Cooling: Cold plates directly on GPUs/CPUs
- Examples: Next-gen hyperscale deployments

**Our Target: PUE < 1.20**

Achieving this requires:
1. **Liquid cooling** (direct-to-chip or rear-door heat exchangers)
2. **AI-driven cooling optimization** (ProphetStor Smart Cooling)
3. **Free cooling** when ambient temperature permits
4. **High-efficiency power distribution** (minimal conversion losses)

#### PUE Breakdown: Where Power Goes

For a **PUE 1.20** facility:

```
Total Facility Power: 342.1 MW (350,000 GPUs)

Power Category              Power Draw    Percentage
──────────────────────────────────────────────────────
IT Equipment                285.0 MW      83.3%
├─ GPUs                     245.0 MW      71.6%
├─ CPUs, memory, network    32.5 MW       9.5%
└─ Switches, storage        7.5 MW        2.2%

Cooling Systems             45.0 MW       13.2%
├─ Chillers                 25.0 MW       7.3%
├─ Pumps                    12.0 MW       3.5%
└─ Cooling towers           8.0 MW        2.3%

Power Distribution          8.5 MW        2.5%
├─ Transformer losses       4.0 MW        1.2%
├─ UPS systems              3.0 MW        0.9%
└─ PDU efficiency           1.5 MW        0.4%

Facilities (Lights, HVAC)   3.6 MW        1.1%
──────────────────────────────────────────────────────
TOTAL                       342.1 MW      100.0%

PUE = 342.1 / 285.0 = 1.20
```

**Optimization Opportunities:**
- AI-driven cooling: 30% reduction in cooling power (ProphetStor data)
- Free cooling: Additional 10-15% reduction when outdoor temperature < 15°C
- High-efficiency transformers: 99%+ efficiency reduces distribution losses

### 1.3 Phased Deployment Model

Deploying 350,000-1,000,000 GPUs simultaneously is logistically impossible and financially imprudent. A phased approach manages risk, validates architecture, and aligns capital deployment with revenue generation.

#### Phase 1: Initial Deployment (100,000 GPUs, 18-24 months)

**Objectives:**
- Validate end-to-end architecture
- Establish operational procedures
- Begin revenue generation from production workloads
- Identify and resolve bottlenecks at 100K scale

**Power Requirements:**
- IT equipment power: 81.4 MW
- Facility power (PUE 1.20): 97.7 MW
- Contracted capacity: ~150 MW per site (with redundancy)

**Distribution:**
- Site A: 50,000 GPUs (60 MW facility power)
- Site B: 30,000 GPUs (36 MW facility power)
- Site C: 20,000 GPUs (24 MW facility power)

**Milestones:**
- Month 0-6: Site preparation, utility coordination
- Month 6-12: Electrical infrastructure construction
- Month 12-18: IT equipment installation
- Month 18-24: Testing, validation, initial production workloads

#### Phase 2: Expansion to Target (350,000 GPUs, 24-36 months)

**Objectives:**
- Scale validated architecture to full Phase 1 target
- Achieve >50% MFU across all sites
- Optimize power consumption and PUE

**Power Requirements:**
- IT equipment power: 285.0 MW
- Facility power (PUE 1.20): 342.1 MW
- Contracted capacity: ~550-700 MW per site

**Distribution:**
- Site A: 140,000 GPUs (168 MW facility power)
- Site B: 140,000 GPUs (168 MW facility power)
- Site C: 70,000 GPUs (84 MW facility power)

**Milestones:**
- Month 24-30: Additional electrical infrastructure
- Month 30-36: GPU installation and network expansion
- Month 36+: Full production operation

#### Phase 3: Future Expansion (700,000-1,000,000 GPUs, 36-60 months)

**Objectives:**
- Utilize full 5GW contracted capacity
- Support next-generation GPUs (B100, MI400, etc.)
- Enable multiple trillion-parameter training runs simultaneously

**Power Requirements:**
- IT equipment power: 570-815 MW
- Facility power (PUE 1.20): 684-978 MW
- Full utilization of contracted 5GW capacity

**Distribution (1M GPU scenario):**
- Site A: 400,000 GPUs (480 MW facility power)
- Site B: 400,000 GPUs (480 MW facility power)
- Site C: 200,000 GPUs (240 MW facility power)

#### Deployment Velocity: Lessons from Meta

Meta's deployment of 350,000 H100 GPUs by end of 2024 demonstrates achievable velocity:
- **Timeline**: ~18-24 months for full deployment
- **Approach**: Multiple datacenter sites operating in parallel
- **Challenges**: GPU allocation, network fabric scaling, operational training

**Our Phase 1 Deployment Rate:**
- 100,000 GPUs over 24 months = 4,167 GPUs/month average
- Peak installation rate: 6,000-8,000 GPUs/month (months 12-18)
- Assumes 3 parallel deployment sites

**Phase 2 Deployment Rate:**
- Additional 250,000 GPUs over 12 months = 20,833 GPUs/month
- Requires established processes, trained teams, validated supply chain

---

## 2. Electrical Infrastructure Design

### 2.1 Utility Coordination and Contracts

Securing gigawatt-scale power delivery is a multi-year process requiring deep coordination with utility providers, regulators, and grid operators.

#### Lead Times and Critical Path

**Utility Coordination Timeline:**
```
Activity                           Duration    Critical Dependencies
───────────────────────────────────────────────────────────────────────
Initial utility discussions        3-6 months  Site selection, power requirements
Utility capacity assessment        2-4 months  Grid capacity studies
Environmental impact studies       6-12 months Local regulations, permitting
Utility approval and contracts     6-12 months Legal, financial terms
Substation design and procurement  12-18 months Equipment lead times
Substation construction            12-24 months Weather, permitting delays
Grid interconnection               3-6 months  Utility coordination
Testing and commissioning          2-4 months  Safety inspections
───────────────────────────────────────────────────────────────────────
TOTAL CRITICAL PATH                36-48 months (3-4 years)
```

**Key Insight**: Electrical infrastructure is often the **longest-lead-time item** in datacenter deployment. Begin utility discussions immediately upon site selection.

#### Power Purchase Agreements (PPAs)

**Contracted Capacity vs. Actual Draw:**

Utilities charge for both:
1. **Demand charges**: Based on peak power draw ($/kW-month)
2. **Energy charges**: Based on total consumption ($/kWh)

For a 2.0 GW site, typical contract structure:

```
Contracted Capacity: 2,000 MW
├─ Firm capacity (guaranteed available): 1,800 MW
├─ Interruptible capacity (can be curtailed): 200 MW
└─ Peak demand charge: $10-15/kW-month

Typical utilization: 70-85%
├─ Average power draw: 1,400-1,700 MW
├─ Peak power draw: 1,800 MW (during full training runs)
└─ Energy charge: $0.03-0.05/kWh (negotiated rate for large load)
```

**Annual Costs for 2 GW Site (80% utilization):**
- Demand charges: 2,000,000 kW × $12/kW-month × 12 months = **$288M/year**
- Energy charges: 1,600 MW × 8,760 hours × $0.04/kWh = **$560M/year**
- **Total**: ~$850M/year per 2 GW site

**Cost Optimization Strategies:**
- **Interruptible rates**: 15-25% discount for curtailable load
- **Time-of-use pricing**: Run workloads during off-peak hours
- **Power factor correction**: Avoid reactive power penalties
- **Renewable energy credits**: Offset carbon footprint, potential tax benefits

#### Grid Integration Requirements

Large datacenters are significant grid loads requiring careful integration:

**Technical Requirements:**
1. **Power factor**: Maintain >0.95 to avoid penalties
2. **Harmonic distortion**: Total Harmonic Distortion (THD) < 5%
3. **Voltage regulation**: ±5% of nominal voltage
4. **Frequency stability**: ±0.1 Hz of 60 Hz (North America)

**Grid Services (Demand Response Programs):**

Datacenters can provide valuable grid services:
- **Load shedding**: Reduce power during grid emergencies (compensated at premium rates)
- **Frequency regulation**: Adjust load to stabilize grid frequency
- **Capacity reserves**: Guaranteed curtailment capability during peak demand

**Revenue potential**: $10-25/kW-year for participation in demand response programs

### 2.2 Substation Design and Transformer Sizing

Substations step down utility transmission voltage (115kV-230kV) to datacenter distribution voltage (typically 13.8kV or 34.5kV).

#### Substation Architecture for 2 GW Site

**Primary Substation Configuration:**
```
Utility Feed (230 kV)
│
├─── Circuit Breaker (2,500 MVA interrupting capacity)
│
├─── Primary Substation
│    ├─ Main Transformer #1: 230kV → 13.8kV (1,000 MVA)
│    ├─ Main Transformer #2: 230kV → 13.8kV (1,000 MVA)
│    └─ Main Transformer #3: 230kV → 13.8kV (500 MVA) [N+1 redundancy]
│
├─── Medium Voltage Distribution (13.8 kV)
│    ├─ Datacenter Building #1 (500 MW)
│    ├─ Datacenter Building #2 (500 MW)
│    ├─ Datacenter Building #3 (500 MW)
│    └─ Datacenter Building #4 (500 MW)
│
└─── Emergency Backup
     ├─ Diesel Generators (200 MW emergency capacity)
     └─ Battery UPS Systems (2N configuration, 15-minute runtime)
```

**Transformer Specifications:**

For 1,000 MVA primary transformer:
- **Voltage ratio**: 230kV / 13.8kV
- **Cooling**: Oil-immersed with forced oil and air cooling (OFAF)
- **Efficiency**: 99.5%+ at rated load
- **Size**: ~40 feet long, 15 feet wide, 20 feet tall
- **Weight**: ~300,000 lbs (150 tons)
- **Lead time**: 18-24 months
- **Cost**: $8-12 million each

#### Secondary Substations (Building-Level)

Each datacenter building receives medium-voltage power (13.8kV) and steps down to low-voltage (480V) for distribution to racks:

**Building Substation (500 MW capacity):**
```
Medium Voltage Feed (13.8 kV)
│
├─── Building Circuit Breaker
│
├─── Secondary Transformers (8× 75 MVA units)
│    ├─ 13.8kV → 480V
│    ├─ Efficiency: 98.5%+
│    └─ N+1 configuration (7 operational, 1 standby)
│
├─── Low Voltage Distribution (480V)
│    ├─ Switchgear and busway
│    ├─ Automatic transfer switches
│    └─ Power distribution units (PDUs)
│
└─── Rack Power Distribution
     ├─ 480V → 208V (rack PDUs)
     └─ GPU servers (208V 3-phase)
```

**Losses in Distribution Chain:**

```
Component                   Input Power    Efficiency    Output Power    Loss
──────────────────────────────────────────────────────────────────────────────
Utility feed                2,000 MW       -             2,000 MW        -
Primary transformer         2,000 MW       99.5%         1,990 MW        10 MW
Medium voltage distribution 1,990 MW       99.8%         1,986 MW        4 MW
Secondary transformers      1,986 MW       98.5%         1,956 MW        30 MW
Low voltage distribution    1,956 MW       99.0%         1,936 MW        20 MW
Rack PDUs                   1,936 MW       97.0%         1,878 MW        58 MW
──────────────────────────────────────────────────────────────────────────────
TOTAL DISTRIBUTION LOSS                                                 122 MW (6.1%)
```

**Optimization**: Using high-efficiency transformers (99.7% vs 99.5%) and improved PDUs (98% vs 97%) reduces losses by ~35 MW, saving **$12M annually** at $0.04/kWh.

### 2.3 UPS Systems and Redundancy

Uninterruptible Power Supply (UPS) systems provide clean, conditioned power and bridge the gap between utility failure and generator startup.

#### 2N UPS Configuration

**Design Philosophy**: 2N (two times N) provides complete redundancy—two fully independent power paths, each capable of handling 100% of the load.

```
Datacenter Building (500 MW IT load)

POWER PATH A (500 MW capacity)          POWER PATH B (500 MW capacity)
│                                        │
├─ Utility Feed A (13.8kV)              ├─ Utility Feed B (13.8kV)
│  └─ From independent substation       │  └─ From independent substation
│                                        │
├─ Diesel Generators A (100 MW)         ├─ Diesel Generators B (100 MW)
│  └─ N+1 configuration                 │  └─ N+1 configuration
│                                        │
├─ Automatic Transfer Switch A          ├─ Automatic Transfer Switch B
│  └─ Switches to generator in <10s     │  └─ Switches to generator in <10s
│                                        │
├─ UPS System A (500 MW, 15 min)        ├─ UPS System B (500 MW, 15 min)
│  └─ Flywheel or battery technology    │  └─ Flywheel or battery technology
│                                        │
└─ PDUs feeding ODD racks (1,3,5...)    └─ PDUs feeding EVEN racks (2,4,6...)
```

**Rack-Level Dual Power:**

Each server has dual power supplies, drawing from both Path A and Path B:
```
GPU Server (9.2 kW)
├─ PSU #1 (4.6 kW) ← Path A
└─ PSU #2 (4.6 kW) ← Path B

Failure scenarios:
├─ Path A fails → PSU #2 handles full 9.2 kW load
├─ Path B fails → PSU #1 handles full 9.2 kW load
└─ Both paths fail → Server shuts down (requires dual-path failure = extremely rare)
```

**UPS Runtime Sizing:**

```
Scenario                    UPS Runtime    Rationale
─────────────────────────────────────────────────────────────────────
Immediate generator start   0 minutes      No UPS needed (risky)
Generator start + paralleling 5 minutes    Minimum viable
Generator start + validation 15 minutes    Industry standard
Extended runtime            30-60 minutes  Allows orderly shutdown
```

**Our specification: 15-minute runtime at full load**

For 500 MW datacenter building:
- UPS capacity per path: 500 MW × 15 min = 125 MWh
- Battery technology: Lithium-ion (higher energy density than lead-acid)
- Flywheel alternative: Lower maintenance, unlimited cycles, but higher cost

**UPS System Costs:**

```
Component                          Quantity    Unit Cost      Total Cost
─────────────────────────────────────────────────────────────────────────
500 MW UPS modules (2N)            20 units    $15M/unit      $300M
Battery banks (15 min runtime)     2 systems   $80M/system    $160M
Installation and commissioning     -           -              $40M
─────────────────────────────────────────────────────────────────────────
TOTAL PER 500 MW BUILDING                                     $500M
```

### 2.4 Backup Power and Generators

Diesel generators provide long-duration backup power when utility feeds fail.

#### Generator Sizing and Configuration

**Requirement**: Support critical IT load during extended utility outages

For 500 MW datacenter building:
```
Critical Load Breakdown             Power Draw
──────────────────────────────────────────────
IT Equipment (GPUs, servers)        425 MW (85% of total)
Cooling (chillers, pumps)           60 MW (12% of total)
Lighting, safety, controls          15 MW (3% of total)
──────────────────────────────────────────────
TOTAL CRITICAL LOAD                 500 MW
```

**Generator Configuration: N+1 Redundancy**
```
Generator Farm (per building)
├─ Generator Unit #1: 25 MW (diesel)
├─ Generator Unit #2: 25 MW (diesel)
├─ Generator Unit #3: 25 MW (diesel)
├─ Generator Unit #4: 25 MW (diesel)
├─ Generator Unit #5: 25 MW (diesel) [Standby]
└─ Total capacity: 125 MW (5× 25 MW units)

Load scenarios:
├─ Normal: 0 MW (utility power)
├─ Utility failure: 100 MW (4 generators, 1 standby)
├─ Generator failure: 100 MW (3 generators + bring standby online)
```

**Why only 100 MW backup for 500 MW building?**

Not all workloads require diesel backup:
- **Tier 1 (Critical - 20%)**: Training checkpoints, model serving (100 MW)
- **Tier 2 (Important - 30%)**: Graceful shutdown, save state (150 MW)
- **Tier 3 (Best-effort - 50%)**: Immediate shutdown acceptable (250 MW)

During utility failure:
1. **Immediate**: UPS maintains all loads (15 minutes)
2. **0-2 minutes**: Generators start and parallel
3. **2-15 minutes**: Tier 3 workloads gracefully terminate, free up 250 MW
4. **15+ minutes**: Generators support 100 MW critical load indefinitely

**Fuel Storage and Runtime:**

```
25 MW Diesel Generator
├─ Fuel consumption: ~200 gallons/hour at full load
├─ Fuel tank: 20,000 gallons (100 hours runtime)
└─ Refueling: Coordinate with fuel delivery during extended outages

Building Generator Farm (4× 25 MW at 80% load)
├─ Total consumption: 640 gallons/hour
├─ Total fuel storage: 80,000 gallons
└─ Runtime: ~125 hours (5+ days) before refueling
```

**Generator Costs:**

```
Component                       Quantity    Unit Cost      Total Cost
────────────────────────────────────────────────────────────────────────
25 MW diesel generators         5 units     $8M/unit       $40M
Fuel storage tanks (20K gal ea) 5 tanks     $500K/tank     $2.5M
Switchgear and paralleling      1 system    $5M            $5M
Installation and commissioning  -           -              $7.5M
────────────────────────────────────────────────────────────────────────
TOTAL PER 500 MW BUILDING                                  $55M
```

### 2.5 Power Distribution Architecture

#### Three-Phase Power Distribution

GPU servers require three-phase power for efficiency and capacity:

```
Three-Phase 208V Distribution

Phase A ─────┬─────> PSU #1 (120V Line-Neutral or 208V Line-Line)
             │
Phase B ─────┼─────> PSU #2 (balanced across phases)
             │
Phase C ─────┴─────> Balanced distribution critical for transformer efficiency

Per-rack power: 40-60 kW (typical)
├─ 5× 8-GPU servers @ 9.2 kW each = 46 kW
└─ Network switches, PDU overhead = 4 kW

Current draw: 46,000W / (208V × √3) = 128 Amps per phase
Maximum: 200A circuit breaker (41.6 kW at 208V 3-phase)
```

#### Rack PDU Architecture

**Intelligent Rack PDU (Power Distribution Unit):**
```
Rack PDU Specifications
├─ Input: 208V 3-phase, 200A (41.6 kW max)
├─ Output: 36× C13 outlets, 6× C19 outlets
├─ Monitoring: Per-outlet current, voltage, power factor
├─ Remote management: SNMP, Redfish API
├─ Efficiency: 97-98%
└─ Cost: $2,000-3,000 per PDU

Per-rack configuration (dual-PDU for redundancy):
├─ PDU A: Connected to Power Path A
├─ PDU B: Connected to Power Path B
└─ Each server draws from both PDUs (dual PSUs)
```

#### Busway Distribution

For high-density deployments, overhead busway provides flexible, high-capacity power distribution:

```
Overhead Busway System

Substation (480V, 3-phase)
│
└─── Main Busway (2,000A capacity)
     │
     ├─── Tap-off #1 → Rack Row 1 (200A)
     ├─── Tap-off #2 → Rack Row 2 (200A)
     ├─── Tap-off #3 → Rack Row 3 (200A)
     ├─── ...
     └─── Tap-off #N → Rack Row N (200A)

Advantages:
├─ Flexible tap-off points (easily reconfigure)
├─ Lower installation cost vs. conduit
├─ Higher reliability than cables
└─ Hot-swappable tap-off boxes
```

**Busway Costs**: $150-250/amp-foot installed (480V busway with tap-offs)

### 2.6 Power Quality and Conditioning

High-performance computing requires clean, stable power:

#### Power Quality Metrics

**Critical Parameters:**
```
Parameter                  Specification       Why It Matters
───────────────────────────────────────────────────────────────────
Voltage stability          ±5% of nominal      Prevents server PSU errors
Frequency stability        ±0.1 Hz (60 Hz)     Critical for synchronization
Power factor               >0.95               Avoids utility penalties
Total Harmonic Distortion  <5% (voltage)       Prevents equipment damage
                           <20% (current)
Transient overvoltage      <140% nominal       Protects semiconductors
Interruption duration      <16ms (1 cycle)     Servers tolerate briefly
```

#### Power Factor Correction

**The Problem**: Non-linear loads (server PSUs, VFDs) draw current that lags or leads voltage, creating reactive power.

**Power Factor (PF) = Real Power / Apparent Power**

```
Example: 1,000 kW load with PF = 0.85
├─ Real power: 1,000 kW
├─ Apparent power: 1,000 kW / 0.85 = 1,176 kVA
└─ Utility charges for 1,176 kVA, but only 1,000 kW does useful work

With PF correction to 0.98:
├─ Real power: 1,000 kW
├─ Apparent power: 1,000 kW / 0.98 = 1,020 kVA
└─ Saves 156 kVA in utility capacity charges
```

**Active Power Factor Correction (PFC):**

Modern server PSUs include active PFC:
- Power factor: 0.95-0.99
- Harmonic distortion: <5%
- Compliance: 80 PLUS Titanium (96%+ efficiency)

**Facility-Level Correction:**

For large installations, deploy capacitor banks or active filters:
```
Component                   Cost/kVAR      500 MW Building
──────────────────────────────────────────────────────────────
Capacitor banks             $20-40         $5-10M
Active harmonic filters     $100-150       $25-35M
Maintenance (annual)        5% of capital  $1.5-2M/year
```

---

## 3. Multi-Site Power Distribution

### 3.1 Site Selection and Power Allocation

The 5GW total capacity is distributed across three sites based on power availability, construction timelines, and strategic considerations.

#### Site A: Primary East Coast (2.0 GW)

**Location Profile:**
```
Geographic Region:          Ohio / Pennsylvania corridor
Utility Provider:           Multiple providers (diversification)
Contracted Capacity:        2,000 MW
Current Buildout (Phase 1): 140,000-160,000 H100 GPUs
Future Expansion (Phase 2): Up to 400,000 GPUs

Power Infrastructure:
├─ Primary substation: 2× 1,000 MVA transformers + 1× 500 MVA (N+1)
├─ Datacenter buildings: 4× 500 MW buildings
├─ UPS capacity: 2N configuration, 15-minute runtime
└─ Generator backup: 100 MW per building (400 MW total)

Power Density:
├─ Per building: 500 MW / 40,000 GPUs = 12.5 kW per GPU
├─ Per rack: 40-50 kW (5× 8-GPU servers)
└─ Per square foot: 400-500 W/sq ft (high-density zone)
```

**Why Ohio/Pennsylvania?**
- Abundant utility power capacity (proximity to coal/nuclear plants)
- Low electricity costs ($0.035-0.045/kWh industrial rates)
- Favorable climate for free cooling (moderate summers, cold winters)
- Proximity to fiber optic backbone (Chicago-NYC corridor)
- Tax incentives for datacenter investment

#### Site B: Primary Central (2.0 GW)

**Location Profile:**
```
Geographic Region:          Iowa / Nebraska / Kansas
Utility Provider:           Municipal utilities + renewables
Contracted Capacity:        2,000 MW
Current Buildout (Phase 1): 140,000-160,000 H100 GPUs
Future Expansion (Phase 2): Up to 400,000 GPUs

Power Infrastructure:
├─ Primary substation: 2× 1,000 MVA transformers + 1× 500 MVA (N+1)
├─ Datacenter buildings: 4× 500 MW buildings
├─ UPS capacity: 2N configuration, 15-minute runtime
└─ Generator backup: 100 MW per building (400 MW total)

Renewable Energy Integration:
├─ On-site wind: 200 MW capacity (30% capacity factor = 60 MW avg)
├─ On-site solar: 100 MW capacity (25% capacity factor = 25 MW avg)
├─ Grid renewables: 40% wind/solar in regional grid mix
└─ Carbon neutrality: 95%+ through renewable credits
```

**Why Iowa/Nebraska?**
- Lowest electricity costs in US ($0.025-0.035/kWh)
- Abundant wind power (Iowa: 60%+ renewable grid)
- Excellent cooling climate (free cooling 6+ months/year)
- Available land for large campus development
- Google precedent (significant datacenter presence)

#### Site C: Secondary West Coast (1.0 GW)

**Location Profile:**
```
Geographic Region:          Oregon / Washington
Utility Provider:           Hydroelectric (BPA, public utilities)
Contracted Capacity:        1,000 MW
Current Buildout (Phase 1): 70,000-80,000 H100 GPUs
Future Expansion (Phase 2): Up to 200,000 GPUs

Power Infrastructure:
├─ Primary substation: 1× 1,000 MVA transformer + 1× 250 MVA (N+1)
├─ Datacenter buildings: 2× 500 MW buildings
├─ UPS capacity: 2N configuration, 15-minute runtime
└─ Generator backup: 100 MW per building (200 MW total)

Strategic Advantages:
├─ 100% carbon-free power (hydroelectric + wind)
├─ Low electricity costs ($0.030-0.040/kWh)
├─ Cool, stable climate (minimal cooling load)
└─ Proximity to Pacific Rim (latency optimization for Asia)
```

**Why Pacific Northwest?**
- 100% renewable hydroelectric base load
- Lowest PUE achievable (free cooling year-round)
- Low seismic risk compared to California
- Meta, Amazon precedent (Prineville, OR; The Dalles, OR)

### 3.2 Load Balancing and Redundancy

Multi-site deployment provides both capacity scaling and resilience:

#### Geographic Redundancy

```
Failure Scenario Planning

Single Building Failure (500 MW):
├─ Impact: 10% of total capacity offline
├─ Mitigation: Workloads failover to other buildings in same site
├─ Recovery time: <30 minutes (automated)
└─ Training impact: <2% (checkpoint restore + resume)

Single Site Failure (2,000 MW):
├─ Impact: 40% of total capacity offline (Site A or B)
├─ Mitigation: Workloads redistribute to Site B/C or Site A/C
├─ Recovery time: 2-4 hours (cross-datacenter checkpoint transfer)
└─ Training impact: 5-10% (reduced parallelism, re-sharding)

Multi-Site Partial Failure (Grid Event):
├─ Scenario: Regional grid instability affects Sites A + C
├─ Impact: 60% capacity degraded (generators sustain critical load)
├─ Mitigation: Site B operates normally, A/C run critical workloads only
└─ Duration: 2-24 hours (until grid stabilizes)
```

#### Dynamic Load Balancing

**ProphetStor Federator.ai** orchestrates workload placement across sites based on:

1. **Power availability**: Route workloads to sites with excess capacity
2. **Energy cost**: Shift batch jobs to sites with lowest current electricity rates
3. **Thermal conditions**: Move workloads from thermally constrained sites
4. **Network latency**: Place tightly-coupled jobs in same site

**Example Scenario**:
```
Time: 2:00 PM, Summer Day

Site A (Ohio):
├─ Current load: 1,600 MW / 2,000 MW (80%)
├─ Electricity rate: $0.12/kWh (peak demand period)
├─ Thermal margin: 15% (outdoor temp 32°C)
└─ Decision: Shift new batch jobs to Site B

Site B (Iowa):
├─ Current load: 1,400 MW / 2,000 MW (70%)
├─ Electricity rate: $0.04/kWh (off-peak, wind surplus)
├─ Thermal margin: 40% (outdoor temp 24°C)
└─ Decision: Accept workloads from Site A

Site C (Oregon):
├─ Current load: 800 MW / 1,000 MW (80%)
├─ Electricity rate: $0.03/kWh (constant hydro)
├─ Thermal margin: 50% (outdoor temp 18°C)
└─ Decision: Maintain current allocation

Savings: 200 MW × 8 hours × ($0.12 - $0.04)/kWh = $128,000 saved
```

### 3.3 Inter-Site Power Considerations

While power itself is not transferred between sites, coordination is essential:

#### Demand Response Coordination

**Grid Operator Signals** (ISO/RTO):

During grid emergencies, datacenters participate in demand response:
```
Event Type              Site Response           Compensation
─────────────────────────────────────────────────────────────────
Peak shaving            Reduce 10% load         $500/MWh
Emergency DR            Reduce 30% load         $2,000-5,000/MWh
Frequency regulation    ±5% load modulation     $50/MW-year capacity

Annual revenue potential: $10-25M across 5 GW portfolio
```

**Coordinated Response:**
- Site A reduces 200 MW → Site B/C increase 100 MW each (maintain total training throughput)
- DiLoCo training architecture tolerates temporary site capacity reductions
- Revenue from demand response offsets electricity costs

#### Time-Zone Arbitrage

**Follow-the-Sun Power Optimization:**
```
Time (UTC)  Site A (EST)  Site B (CST)  Site C (PST)  Strategy
────────────────────────────────────────────────────────────────
00:00       7 PM (off-pk) 6 PM (off-pk) 5 PM (off-pk) High utilization all sites
06:00       1 AM (min)    12 AM (min)   11 PM (min)   Lowest cost: max all sites
12:00       7 AM (ramp)   6 AM (ramp)   5 AM (ramp)   Reduce east, boost west
18:00       1 PM (peak)   12 PM (peak)  11 AM (peak)  Shift load to off-peak sites
```

**Annual Savings**: $15-30M through time-zone aware workload scheduling

---

## 4. Power Optimization Strategies

### 4.1 ProphetStor AI-Driven Power Management

ProphetStor Federator.ai provides ML-driven optimization of power consumption across the infrastructure.

#### Core Capabilities

**1. Workload-Aware Power Scheduling**

```
Traditional Scheduler              ProphetStor Federator.ai
─────────────────────────────────────────────────────────────────
├─ Static resource allocation      ├─ Dynamic ML-based prediction
├─ No power awareness              ├─ Real-time power optimization
├─ Manual intervention required    ├─ Automated decision-making
└─ 60-70% resource utilization     └─ 85-95% resource utilization

Power savings demonstrated: 15-25% reduction in wasted capacity
```

**2. Thermal-Aware Job Placement**

Federator.ai monitors thermal conditions and places workloads to minimize cooling energy:

```
Input Data (Real-Time):
├─ Per-rack temperature sensors (inlet, outlet)
├─ Chiller COP (Coefficient of Performance)
├─ Outdoor air temperature (for free cooling optimization)
└─ GPU utilization and power draw

ML Model Predicts:
├─ Future thermal load per rack/zone
├─ Cooling system efficiency at different loads
└─ Optimal job placement to minimize cooling energy

Example:
├─ Rack A: 45 kW load, inlet temp 22°C, outlet 32°C → OK
├─ Rack B: 48 kW load, inlet temp 26°C, outlet 38°C → THERMAL STRESS
└─ Decision: Place new job in Rack A, migrate workload from Rack B
```

**Measured Impact**: 30% reduction in cooling energy (ProphetStor case studies)

**3. Predictive Power Budgeting**

LSTM models forecast power consumption patterns:

```
Prediction Horizon: 15 minutes to 24 hours

Inputs:
├─ Historical power draw patterns (1 year of data)
├─ Scheduled job queue (batch vs. interactive workloads)
├─ Time-of-day, day-of-week seasonality
└─ Outdoor temperature forecast (impacts cooling load)

Outputs:
├─ Expected peak power draw (±3% accuracy)
├─ Optimal generator pre-positioning (fuel efficiency)
└─ Grid demand response participation decisions

Value:
├─ Avoid utility demand charge penalties (5-10% savings)
├─ Optimize generator fuel consumption (30% reduction in waste)
└─ Enable proactive thermal management
```

#### Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│          ProphetStor Federator.ai Control Plane             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │ ML Prediction│   │ Optimization │   │  Execution   │    │
│  │    Engine    │──>│    Engine    │──>│   Engine     │    │
│  └──────────────┘   └──────────────┘   └──────────────┘    │
│         │                    │                   │           │
└─────────┼────────────────────┼───────────────────┼───────────┘
          │                    │                   │
          ▼                    ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Data Sources  │  │ Constraints     │  │ Actuators       │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│• Power meters   │  │• Power budgets  │  │• Job scheduler  │
│• Temp sensors   │  │• Thermal limits │  │• DVFS controls  │
│• GPU telemetry  │  │• SLA targets    │  │• Cooling system │
│• Electricity    │  │• Cost targets   │  │• Power limiters │
│  price feeds    │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### 4.2 Dynamic Voltage and Frequency Scaling (DVFS)

Modern GPUs support power-performance trade-offs through DVFS:

#### NVIDIA GPU Power Management

**Power States (H100):**
```
State   Description              Power Draw    Performance    Use Case
────────────────────────────────────────────────────────────────────────
P0      Max performance          700W (100%)   100%           Training compute
P1      Slightly reduced         600W (86%)    95%            Memory-bound ops
P2      Balanced                 500W (71%)    85%            Communication phases
P8      Idle                     50W (7%)      N/A            No workload
```

**Dynamic Power Capping:**

```
nvidia-smi -pl 500  # Set power limit to 500W (from 700W default)

Impact:
├─ Power: 29% reduction (700W → 500W)
├─ Performance: ~10-15% reduction (depends on workload)
└─ ROI: Power-limited during communication phases → minimal training impact

Example: All-reduce operation (communication-bound)
├─ At 700W: GPUs idle waiting for network, wasted power
├─ At 500W: GPUs throttled but still waiting for network, no performance loss
└─ Savings: 200W per GPU during 20-30% of training time
```

**Workload-Aware DVFS:**

ProphetStor Federator.ai detects training phases and adjusts power limits:

```
Training Phase              GPU State    Power Limit    Rationale
─────────────────────────────────────────────────────────────────────
Forward pass (compute)      P0           700W           Maximize FLOPs
Backward pass (compute)     P0           700W           Maximize FLOPs
Gradient all-reduce (comm)  P2           500W           Network-bound, save power
Optimizer step (memory)     P1           600W           Memory-bound
Checkpoint save (I/O)       P8           50W            Idle during blocking I/O
```

**Annual Savings (350,000 GPUs):**
```
Baseline: 350,000 GPUs × 700W × 8,760 hours × $0.04/kWh = $860M/year

With DVFS (15% average reduction):
├─ Power: 350,000 × 595W avg × 8,760 × $0.04 = $731M/year
└─ Savings: $129M/year (15% reduction)

Training impact: <3% (validated in production at MegaScale)
```

### 4.3 Workload-Aware Power Scheduling

Not all training jobs have equal power characteristics:

#### Job Classification by Power Profile

```
Job Type                Power Pattern       Priority    Scheduling Strategy
──────────────────────────────────────────────────────────────────────────
Initial training        High, sustained     P1          Schedule during off-peak
Continued training      High, sustained     P1          Maintain continuity
Fine-tuning            Medium, bursty      P2          Fill gaps, interruptible
Evaluation             Low, intermittent   P3          Run during peak hours
Inference serving      Medium, variable    P0          Critical, 24/7
Checkpointing          Spike (I/O)         P1          Schedule overnight
```

**Peak Shaving Strategy:**

```
Example: August weekday, 2 PM (utility peak demand period)

Current load: 4,500 MW across 3 sites
Peak demand charge tier: >4,000 MW costs +$5/kW-month

Options:
1. Continue all jobs → 4,500 MW → $22.5M demand charge
2. Pause P3 jobs → 4,200 MW → $21M demand charge
3. Pause P2+P3 jobs → 3,800 MW → $19M demand charge

Decision: Pause P3 jobs (200 MW)
├─ 2-hour pause costs: <1% training time
├─ Demand charge savings: $1.5M/month ($18M/year)
└─ Jobs resume automatically after peak period
```

### 4.4 Grid Demand Response Programs

Datacenters can monetize flexible power consumption:

#### PJM Capacity Market (Example: Site A in Ohio/Pennsylvania)

**Program Structure:**
```
Commitment: Reduce 500 MW within 30 minutes of grid emergency
Duration: 100 hours/year maximum
Compensation: $150/MW-day capacity payment

Annual Revenue:
├─ 500 MW × $150/MW-day × 365 days = $27.4M/year
├─ Lost training time: ~100 hours × 500 MW = 50 GWh
├─ Opportunity cost: 50,000 MWh × $0.04/kWh = $2M
└─ Net revenue: $25.4M/year
```

**Implementation:**

```
Grid Emergency Event (frequency drops below 59.9 Hz)

Step 1: Automated Signal Reception (5 seconds)
├─ Grid operator sends curtailment signal
└─ Federator.ai receives and validates

Step 2: Workload Classification (30 seconds)
├─ Identify interruptible jobs (P2, P3 priority)
├─ Save checkpoints for resumption
└─ Gracefully terminate 500 MW of load

Step 3: Load Reduction (10 minutes)
├─ 500 MW load shed complete
├─ Verification signal sent to grid operator
└─ Maintain critical workloads (P0, P1)

Step 4: Recovery (when event ends)
├─ Grid operator sends "all clear"
├─ Restore workloads from checkpoints
└─ Resume normal operations (30-60 min)
```

**Advanced: Frequency Regulation**

Participate in real-time frequency balancing:
```
Service: ±50 MW modulation within 4 seconds
Frequency: Continuous monitoring, response 100+ times/day
Compensation: $50/MW-year capacity + energy payments

Technical implementation:
├─ Direct connection to grid frequency signal
├─ Automated load modulation (±50 MW)
├─ GPU power limiting (DVFS) for rapid response
└─ Minimal training impact (<0.5%)

Annual revenue: 50 MW × $50/MW-year = $2.5M
```

### 4.5 Renewable Energy Integration

#### On-Site Solar and Wind (Site B: Iowa)

**Solar Installation:**
```
Capacity: 100 MW DC (80 MW AC after inverter losses)
Annual generation: 100 MW × 25% capacity factor × 8,760 hours = 219 GWh
Cost: $1.20/watt × 100 MW = $120M capital
Lifespan: 25 years

Economics:
├─ Annual generation value: 219,000 MWh × $0.04/kWh = $8.8M
├─ Payback period: 13.6 years
├─ LCOE (Levelized Cost): $0.032/kWh
└─ Carbon offset: 219 GWh × 0.4 kg CO₂/kWh = 87,600 tons CO₂/year
```

**Wind Installation:**
```
Capacity: 200 MW
Annual generation: 200 MW × 35% capacity factor × 8,760 hours = 613 GWh
Cost: $1.50/watt × 200 MW = $300M capital
Lifespan: 20 years

Economics:
├─ Annual generation value: 613,000 MWh × $0.04/kWh = $24.5M
├─ Payback period: 12.2 years
├─ LCOE: $0.028/kWh
└─ Carbon offset: 613 GWh × 0.4 kg CO₂/kWh = 245,200 tons CO₂/year
```

**Grid Integration:**
```
On-site renewables: 80 MW solar + 70 MW wind (avg) = 150 MW
Site B datacenter load: 1,400 MW average
Renewable coverage: 150 / 1,400 = 10.7%

Storage strategy (optional):
├─ Battery storage: 500 MWh (2-hour buffer)
├─ Cost: $300M ($600/kWh installed)
└─ Value: Time-shift renewables to peak demand hours
```

---

## 5. Cost Analysis

### 5.1 Capital Expenditure Breakdown

#### Electrical Infrastructure (Phase 1: 350,000 GPUs)

```
Component                          Per-Site Cost    Sites    Total Cost
─────────────────────────────────────────────────────────────────────────
PRIMARY SUBSTATIONS
├─ Transformers (1,000 MVA)        $30M            6 units   $180M
├─ Switchgear & protection         $20M            3 sites   $60M
├─ Civil works & installation      $15M            3 sites   $45M
└─ SUBTOTAL: Substations                                     $285M

BUILDING ELECTRICAL
├─ Secondary transformers          $25M            12 bldg   $300M
├─ Switchgear & distribution       $20M            12 bldg   $240M
├─ Busway & cabling               $15M            12 bldg   $180M
└─ SUBTOTAL: Distribution                                    $720M

BACKUP POWER
├─ UPS systems (2N, 15 min)        $500M           12 bldg   $6,000M
├─ Diesel generators (100 MW)      $55M            12 bldg   $660M
├─ Fuel storage & transfer         $10M            12 bldg   $120M
└─ SUBTOTAL: Backup Power                                    $6,780M

RACK PDUs & CIRCUITS
├─ Rack PDUs (2 per rack)          $5K             87,500    $438M
├─ Electrical circuits & panels    $3K             87,500    $263M
└─ SUBTOTAL: Rack Distribution                               $701M

INSTALLATION & COMMISSIONING
├─ Labor (electrical contractors)                            $800M
├─ Testing & validation                                      $200M
├─ Project management                                        $150M
└─ SUBTOTAL: Installation                                    $1,150M

─────────────────────────────────────────────────────────────────────────
TOTAL ELECTRICAL CAPEX (Phase 1)                             $9,636M
                                                             ≈ $9.6B

Per-GPU electrical cost: $9.6B / 350,000 = $27,429 per GPU
```

#### Electrical Infrastructure (Phase 2: 1,000,000 GPUs)

```
Additional Infrastructure Required

Transformers & substations          $600M
Building electrical                  $1,500M
UPS systems expansion               $10,000M
Generators                          $800M
Rack distribution                   $1,200M
Installation & commissioning        $2,400M
─────────────────────────────────────────────
TOTAL ADDITIONAL CAPEX              $16.5B

TOTAL ELECTRICAL (Phase 1 + 2):     $26.1B
≈ $15-20B range cited in Chapter 1 (conservative Phase 1.5 scenario)
```

### 5.2 Operational Expenditure (Annual)

#### Electricity Costs (Phase 1: 350,000 GPUs)

**Blended Rate Calculation:**
```
Site A (Ohio): 140,000 GPUs
├─ Average load: 168 MW facility power (PUE 1.20)
├─ Rate: $0.04/kWh blended (demand + energy)
├─ Annual cost: 168 MW × 8,760 hours × $0.04 = $588M

Site B (Iowa): 140,000 GPUs
├─ Average load: 168 MW facility power
├─ Rate: $0.035/kWh (lower cost, renewables)
├─ Annual cost: 168 MW × 8,760 × $0.035 = $515M

Site C (Oregon): 70,000 GPUs
├─ Average load: 84 MW facility power
├─ Rate: $0.038/kWh (hydroelectric)
├─ Annual cost: 84 MW × 8,760 × $0.038 = $280M

─────────────────────────────────────────────
TOTAL ANNUAL ELECTRICITY                $1,383M
```

**Cost Optimization Impact:**
```
Baseline (PUE 1.50, no optimization)    $1,660M/year
With PUE 1.20                          -$280M/year
With DVFS (15% reduction)              -$207M/year
With demand response revenue           +$50M/year
─────────────────────────────────────────────
Optimized annual cost                   $1,223M/year

Savings vs. baseline: $437M/year (26% reduction)
```

#### Facilities Maintenance

```
Category                            Annual Cost
─────────────────────────────────────────────────
Electrical infrastructure M&O       $150M
├─ Transformer maintenance          $30M
├─ UPS battery replacement          $80M (3-year cycle)
├─ Generator testing & fuel         $25M
└─ Switchgear & protective relays   $15M

Cooling systems M&O                 $120M
├─ Chiller maintenance              $50M
├─ Pump & motor service             $30M
├─ Cooling tower chemicals          $25M
└─ Heat exchanger cleaning          $15M

Facilities & grounds                $30M
├─ Security & access control        $10M
├─ Fire suppression testing         $8M
├─ HVAC (office areas)              $7M
└─ Landscaping & grounds            $5M

─────────────────────────────────────────────────
TOTAL FACILITIES MAINTENANCE        $300M/year
```

#### Personnel Costs

```
Role                        Headcount    Avg Salary    Total Cost
───────────────────────────────────────────────────────────────────
Electrical engineers        50           $150K         $7.5M
Facilities engineers        75           $120K         $9.0M
Power system operators      120          $100K         $12.0M
Maintenance technicians     200          $80K          $16.0M
Datacenter managers         12           $200K         $2.4M
Shift supervisors           36           $110K         $4.0M
Safety & compliance         20           $130K         $2.6M
───────────────────────────────────────────────────────────────────
SUBTOTAL: Electrical/Facilities Personnel             $53.5M

Infrastructure architects   30           $180K         $5.4M
Network operations          80           $140K         $11.2M
ML engineering (platform)   100          $200K         $20.0M
Site reliability engineers  60           $180K         $10.8M
Management & admin          50           $150K         $7.5M
───────────────────────────────────────────────────────────────────
TOTAL PERSONNEL COSTS                                 $108.4M/year
```

#### Software and Monitoring

```
System                              Annual Cost
─────────────────────────────────────────────────
ProphetStor Federator.ai            $15M
├─ Enterprise license               $10M
└─ Professional services            $5M

DCIM (Data Center Infrastructure)   $8M
├─ Schneider EcoStruxure or similar
└─ Power monitoring, environmental

GPU monitoring (NVIDIA DCGM)        $3M
├─ Enterprise support
└─ Integration services

Network monitoring                  $5M
├─ Arista/NVIDIA network management
└─ Traffic analysis tools

Orchestration (Kubernetes, Slurm)   $4M
├─ Enterprise support
└─ Custom development

─────────────────────────────────────────────────
TOTAL SOFTWARE & MONITORING         $35M/year
```

### 5.3 Total Cost of Ownership (5-Year)

#### Phase 1: 350,000 GPUs

```
Cost Category                Year 0      Year 1-5 (ea)   5-Year Total
───────────────────────────────────────────────────────────────────────
CAPITAL EXPENDITURE
GPUs (H100 @ $30K each)     $10,500M     -             $10,500M
Electrical infrastructure   $9,600M      -             $9,600M
Network fabric              $4,200M      -             $4,200M
Cooling systems             $3,600M      -             $3,600M
Storage systems             $2,400M      -             $2,400M
Buildings & construction    $6,000M      -             $6,000M
───────────────────────────────────────────────────────────────────────
SUBTOTAL CAPEX              $36,300M                   $36,300M

OPERATIONAL EXPENDITURE
Electricity (optimized)     -            $1,223M       $6,115M
Facilities maintenance      -            $300M         $1,500M
Personnel                   -            $108M         $540M
Network bandwidth (WAN)     -            $75M          $375M
Software & monitoring       -            $35M          $175M
───────────────────────────────────────────────────────────────────────
SUBTOTAL OPEX               -            $1,741M       $8,705M

───────────────────────────────────────────────────────────────────────
TOTAL 5-YEAR TCO                                       $45,005M
                                                       ≈ $45 billion

Per-GPU 5-year TCO: $45B / 350,000 = $128,571 per GPU
Monthly OpEx per GPU: $1,741M / 350,000 / 12 = $414/month
```

#### Cost Sensitivity Analysis

**Electricity Price Impact:**
```
Scenario                    Annual OpEx    5-Year TCO    Δ vs. Base
────────────────────────────────────────────────────────────────────
Base case ($0.04/kWh)       $1,741M        $45.0B        -
Low cost ($0.03/kWh)        $1,486M        $43.7B        -$1.3B (-2.9%)
High cost ($0.06/kWh)       $2,251M        $47.6B        +$2.6B (+5.8%)
```

**PUE Impact:**
```
PUE      Facility Power    Annual Elec    5-Year TCO    Δ vs. 1.20
─────────────────────────────────────────────────────────────────────
1.15     327.6 MW          $1,146M        $44.0B        -$1.0B
1.20     342.1 MW          $1,197M        $45.0B        Baseline
1.30     370.6 MW          $1,296M        $45.5B        +$0.5B
1.50     427.5 MW          $1,495M        $46.5B        +$1.5B
```

**Key Insight**: Every 0.10 reduction in PUE saves **$200M over 5 years** at 350K GPU scale.

### 5.4 Return on Investment (ROI)

Assuming LLM training infrastructure generates revenue through API access, model licensing, and derivative products:

#### Revenue Model (Conservative)

```
Revenue Source              Year 1    Year 2    Year 3    Year 4    Year 5
─────────────────────────────────────────────────────────────────────────
Model API access            $2.0B     $5.0B     $10.0B    $15.0B    $20.0B
Enterprise licensing        $0.5B     $1.5B     $3.0B     $5.0B     $7.0B
Cloud inference services    $0.3B     $1.0B     $2.5B     $4.5B     $7.0B
Data products & insights    $0.2B     $0.5B     $1.0B     $1.5B     $2.0B
─────────────────────────────────────────────────────────────────────────
TOTAL REVENUE               $3.0B     $8.0B     $16.5B    $26.0B    $36.0B

Operating costs             $1.7B     $1.8B     $1.9B     $2.0B     $2.1B
─────────────────────────────────────────────────────────────────────────
EBITDA                      $1.3B     $6.2B     $14.6B    $24.0B    $33.9B

Cumulative cash flow        -$35.0B   -$29.6B   -$16.9B   +$5.2B    +$37.2B
```

**Payback Period**: 3.5 years
**5-Year ROI**: ($37.2B - $36.3B) / $36.3B = **2.4% simple ROI**
**IRR (Internal Rate of Return)**: ~18% (accounting for time value of money)

**Note**: These are conservative estimates. Leading AI companies (OpenAI, Anthropic) demonstrate significantly higher revenue per compute unit.

---

## 6. Risk Assessment and Mitigation

### 6.1 Electrical Infrastructure Risks

#### Risk 1: Utility Capacity Constraints

**Description**: Utility cannot deliver contracted capacity due to transmission limitations or generation shortfalls.

**Likelihood**: Medium (30%)
**Impact**: High (delays 6-18 months)

**Mitigation**:
1. **Early engagement**: Begin utility discussions 36 months before needed
2. **Diverse substations**: Connect to multiple utility feeds
3. **On-site generation**: Deploy 200+ MW of solar/wind to reduce grid dependence
4. **Flexible siting**: Maintain backup site options if primary utility falls through

#### Risk 2: Transformer Lead Time Delays

**Description**: 1,000 MVA transformers have 18-24 month lead times; delays cascade entire project.

**Likelihood**: High (60%)
**Impact**: High (12-month delay)

**Mitigation**:
1. **Early procurement**: Order transformers before site finalization
2. **Standardization**: Use common transformer specs across sites (fungible inventory)
3. **Vendor diversity**: Split orders across GE, Siemens, ABB
4. **Strategic reserves**: Purchase 10% spare transformers (1-2 units) proactively

#### Risk 3: Power Quality Issues

**Description**: Harmonics, voltage sags, or transients damage sensitive IT equipment.

**Likelihood**: Medium (40%)
**Impact**: Medium (equipment damage $10-50M)

**Mitigation**:
1. **Active filters**: Deploy harmonic filtering at substation level
2. **Power factor correction**: Maintain >0.95 PF site-wide
3. **UPS systems**: 2N UPS provides conditioning and isolation
4. **Monitoring**: Real-time power quality monitoring with automated alerts

### 6.2 Cost Overrun Risks

#### Risk 4: Electricity Price Volatility

**Description**: Natural gas prices or grid emergencies cause 50-100% spikes in electricity costs.

**Likelihood**: Medium (40%)
**Impact**: Medium ($500M+ annual increase)

**Mitigation**:
1. **Fixed-rate contracts**: Lock in 70% of load at fixed $0.04/kWh (3-5 year terms)
2. **Renewable PPAs**: Solar/wind PPAs at $0.025-0.030/kWh (20-year fixed)
3. **Demand response**: Generate revenue during peak periods (offset costs)
4. **Workload flexibility**: Shift batch jobs to off-peak hours (30% load is time-flexible)

#### Risk 5: Capital Cost Escalation

**Description**: Labor shortages, material inflation increase electrical infrastructure costs 20-40%.

**Likelihood**: High (70%)
**Impact**: High ($2-4B additional CapEx)

**Mitigation**:
1. **Fixed-price contracts**: Lock in 60% of construction as fixed-price EPC
2. **Early procurement**: Buy long-lead items (transformers, switchgear) upfront
3. **Contingency budgeting**: 25% contingency on electrical infrastructure
4. **Phased deployment**: Validate costs in Phase 1 before committing to Phase 2

---

## 7. Implementation Roadmap

### Month 0-6: Planning and Site Selection

- [ ] Finalize datacenter site selection (Sites A, B, C)
- [ ] Initiate utility discussions and capacity assessments
- [ ] Conduct environmental impact studies
- [ ] Engage electrical engineering firms (Black & Veatch, AECOM, etc.)
- [ ] Develop detailed electrical one-line diagrams
- [ ] Order long-lead transformers (1,000 MVA units)

### Month 6-18: Utility Coordination and Permitting

- [ ] Execute Power Purchase Agreements (PPAs)
- [ ] Obtain environmental permits
- [ ] Finalize utility interconnection agreements
- [ ] Award substation construction contracts
- [ ] Procure UPS systems and generators
- [ ] Award building electrical contracts

### Month 18-30: Construction Phase 1

- [ ] Build primary substations (230kV → 13.8kV)
- [ ] Install primary transformers and switchgear
- [ ] Construct datacenter buildings (civil works)
- [ ] Install building electrical systems (transformers, busway, PDUs)
- [ ] Deploy UPS systems and generator farms
- [ ] Install ProphetStor Federator.ai monitoring

### Month 30-36: Commissioning and Phase 1 Go-Live

- [ ] Electrical testing and commissioning
- [ ] Load testing (100% capacity validation)
- [ ] Install initial 100,000 GPUs
- [ ] Validate end-to-end power delivery
- [ ] Achieve PUE <1.20 target
- [ ] Begin production workloads

### Month 36-48: Phase 2 Expansion

- [ ] Deploy additional 250,000 GPUs
- [ ] Expand electrical infrastructure to full 5GW
- [ ] Optimize power consumption (DVFS, demand response)
- [ ] Achieve >85% utilization across all sites

---

## 8. Key Takeaways

### Critical Success Factors

1. **Early Utility Engagement**: 36-48 month lead time for gigawatt-scale power delivery
2. **PUE <1.20 Target**: Liquid cooling and AI-driven optimization essential for cost control
3. **2N Redundancy**: Dual power paths for 99.99% uptime requirement
4. **Multi-Site Distribution**: Single-site deployments capped at 1.0-1.5 GW; 5GW requires 3+ sites
5. **Power Optimization**: DVFS + demand response saves $200M+ annually

### Validated Design Principles

**From Alibaba HPN**: 18MW building limit = 15,000 GPU capacity
→ Our design: 500 MW building = ~40,000 GPUs (12.5 kW/GPU all-in)

**From Meta (350K H100s)**: Requires ~462 MW facility power at PUE 1.20
→ Our Phase 1: 350K GPUs = 342 MW (achieved through optimization)

**From ProphetStor**: 30% cooling energy reduction through AI-driven management
→ Our design: Integrated Federator.ai from Day 1

### Financial Summary

- **Capital**: $9.6B electrical infrastructure (Phase 1: 350K GPUs)
- **OpEx**: $1.2B/year optimized ($1.7B baseline)
- **5-Year TCO**: $45B total cost of ownership
- **Savings**: $437M/year through PUE optimization + DVFS + demand response

---

## Appendix A: Electrical One-Line Diagram

```
UTILITY TRANSMISSION (230 kV, 2,500 MVA)
│
├─── SITE A: OHIO/PENNSYLVANIA (2.0 GW) ───────────────────┐
│    │                                                      │
│    ├─ Primary Substation #1                              │
│    │  ├─ Transformer T1: 230kV/13.8kV, 1,000 MVA         │
│    │  ├─ Transformer T2: 230kV/13.8kV, 1,000 MVA         │
│    │  └─ Transformer T3: 230kV/13.8kV, 500 MVA (N+1)     │
│    │                                                      │
│    ├─ Building A1 (500 MW)                               │
│    │  ├─ UPS-A (500 MW, 2N config)                       │
│    │  ├─ UPS-B (500 MW, 2N config)                       │
│    │  ├─ Generators: 5× 25 MW (N+1)                      │
│    │  ├─ Secondary Xfmr: 8× 75 MVA (13.8kV → 480V)       │
│    │  └─ Racks: 10,000 racks @ 50 kW each                │
│    │                                                      │
│    ├─ Building A2 (500 MW) [Similar configuration]       │
│    ├─ Building A3 (500 MW) [Similar configuration]       │
│    └─ Building A4 (500 MW) [Similar configuration]       │
│                                                           │
├─── SITE B: IOWA/NEBRASKA (2.0 GW) ───────────────────────┤
│    │                                                      │
│    ├─ Primary Substation #2                              │
│    │  ├─ Transformer T4: 230kV/13.8kV, 1,000 MVA         │
│    │  ├─ Transformer T5: 230kV/13.8kV, 1,000 MVA         │
│    │  └─ Transformer T6: 230kV/13.8kV, 500 MVA (N+1)     │
│    │                                                      │
│    ├─ On-Site Renewables                                 │
│    │  ├─ Wind: 200 MW capacity (70 MW avg)               │
│    │  └─ Solar: 100 MW capacity (25 MW avg)              │
│    │                                                      │
│    ├─ Building B1-B4 (4× 500 MW) [Same as Site A]        │
│    │                                                      │
│                                                           │
└─── SITE C: OREGON/WASHINGTON (1.0 GW) ───────────────────┘
     │
     ├─ Primary Substation #3
     │  ├─ Transformer T7: 230kV/13.8kV, 1,000 MVA
     │  └─ Transformer T8: 230kV/13.8kV, 250 MVA (N+1)
     │
     ├─ Building C1 (500 MW) [Same configuration as above]
     └─ Building C2 (500 MW) [Same configuration as above]

TOTAL CAPACITY: 5,000 MW (5 GW)
TOTAL IT LOAD: 285-815 MW (depending on deployment phase)
CONTRACTED CAPACITY: Allows 900K-1M GPU Phase 2 expansion
```

---

## Appendix B: Power Density Comparison

```
Deployment Era    Technology        Power/Rack    Cooling           PUE
─────────────────────────────────────────────────────────────────────────
2015             CPU servers       5-8 kW        Air (CRAC)        1.6-2.0
2018             GPU (P100/V100)   15-20 kW      Air + containment 1.4-1.6
2021             GPU (A100)        30-40 kW      Rear-door HX      1.3-1.4
2024             GPU (H100)        40-50 kW      Direct-to-chip    1.2-1.3
2025+ (Projected) GPU (B100/GB200) 60-120 kW     Immersion cooling 1.15-1.25

Our Design (2025): 50 kW/rack, Direct-to-chip + rear-door, PUE <1.20
```

---

## Appendix C: Reference Architectures

**Alibaba HPN (2024)**:
- 15,000 GPUs per pod
- 18 MW building power constraint
- 1,200W per GPU all-in (including infrastructure)

**Meta 350K H100 Deployment**:
- 350,000 GPUs total
- ~462 MW facility power at PUE 1.20
- Multi-site deployment (exact distribution undisclosed)

**Google TPU v4 Pods**:
- 4,096 TPUs per pod
- ~10 MW per pod
- PUE: 1.12-1.15 (liquid cooling, free cooling optimization)

---

**End of Chapter 2**

**Next Chapter**: Data Center Site Selection and Multi-State Deployment

---

**Document Version**: 2.0
**Last Updated**: November 15, 2025
**Classification**: Internal - Executive Leadership
**Author**: Infrastructure Planning Team
