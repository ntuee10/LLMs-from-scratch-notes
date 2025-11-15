# Chapter 3: Data Center Site Selection and Multi-State Deployment

**Large-Scale LLM Training Playbook: 1.5 Trillion Parameter Model**
**Target Infrastructure: 5GW Multi-Datacenter Deployment**
**Investment Scale: $100+ Billion**

---

## Executive Summary

The training of a 1.5 trillion parameter language model across 5 gigawatts of infrastructure cannot be accomplished in a single location. This chapter addresses the critical strategic question: *Where do we build?*

Site selection for multi-gigawatt AI infrastructure is fundamentally different from traditional datacenter planning. Power availability becomes the primary constraint—utility companies rarely provision more than 1.5-2.5 GW to a single site. Cooling water access, fiber connectivity, regulatory environments, and construction timelines drive the necessity for 2-3 geographically distributed sites spanning multiple states.

This chapter provides a production-grade framework for site selection validated against real-world deployments by Meta (Virginia, Georgia, Oregon), Google (Ohio, Iowa/Nebraska), and Microsoft (Arizona, Texas, Virginia). We present quantifiable selection criteria, state-by-state analysis, latency optimization strategies, and regulatory compliance requirements necessary for informed decision-making at the board level.

**Key Strategic Imperatives:**

1. **Multi-state deployment is mandatory** due to power delivery limits, regulatory risk concentration, and disaster recovery requirements
2. **2-3 primary sites required** for 5GW total capacity, each sized at 1.5-2.5 GW
3. **Site selection timeline: 24-36 months** from initial assessment to power-on
4. **Geographic distribution must balance** latency (<50ms regional preferred) with regulatory diversity
5. **State incentives can offset** $200M-500M in deployment costs per site

---

## Section 1: Site Selection Criteria

### 1.1 Power Availability: The Primary Constraint

Power availability is the single most critical factor in site selection for multi-gigawatt AI infrastructure. Unlike traditional enterprise datacenters that operate at 5-50 MW scale, training a 1.5 trillion parameter model requires electrical capacity exceeding that of many small cities.

#### Power Requirements by Deployment Phase

**Phase 1 (Initial Deployment): 350,000 H100 GPUs**

| Component | Power Draw | Quantity | Total Power |
|-----------|------------|----------|-------------|
| GPU (H100) | 700W | 350,000 | 245 MW |
| CPU (dual-socket server) | 400W | 43,750 servers | 17.5 MW |
| Memory & motherboard | 100W | 43,750 servers | 4.4 MW |
| Network equipment | 50W | 350,000 endpoints | 17.5 MW |
| Storage infrastructure | - | - | 8 MW |
| **IT Equipment Total** | | | **292 MW** |
| Cooling (PUE 1.20) | | | 58 MW |
| **Total Facility Power** | | | **350 MW** |
| Redundancy & headroom (2×) | | | 350 MW |
| **Contracted Capacity** | | | **700 MW** |

**Phase 2 (Full Build-out): 700,000-1,000,000 GPUs**

- IT power load: 584 MW - 835 MW
- Facility power (PUE 1.20): 701 MW - 1,002 MW
- Contracted capacity with redundancy: **1.4 GW - 2.0 GW**
- **Total across 3 sites: 4.2 GW - 6.0 GW**

This explains why "5GW infrastructure" refers to total contracted electrical capacity across all sites, provisioned for future scaling beyond initial deployment.

#### Utility Power Delivery Constraints

**Single-Site Power Limits**

Based on production deployments and utility engineering constraints:

| Utility Company | Region | Typical Maximum Single-Site Capacity | Lead Time |
|----------------|--------|--------------------------------------|-----------|
| American Electric Power (AEP) | Ohio, Virginia | 1.5 GW - 2.0 GW | 24-36 months |
| Dominion Energy | Virginia | 1.0 GW - 1.5 GW | 18-30 months |
| Duke Energy | North Carolina, South Carolina | 800 MW - 1.2 GW | 24-36 months |
| Georgia Power | Georgia | 1.0 GW - 1.5 GW | 24-30 months |
| PacifiCorp | Oregon | 500 MW - 1.0 GW | 30-42 months |
| NV Energy | Nevada | 800 MW - 1.2 GW | 24-36 months |
| MidAmerican Energy | Iowa | 1.0 GW - 1.5 GW | 24-30 months |

**Key Insights:**

- **1.5-2.5 GW per site is the practical maximum** for utility provisioning
- **No single utility will provision 5 GW to one location** due to grid stability, infrastructure investment, and risk management
- **Lead times of 24-36 months are standard** for substation construction and transmission line upgrades
- **Commitment and deposits required early**: Utilities need 18-24 month advance notice for capacity planning

#### Electrical Infrastructure Requirements

**Substation and Transmission**

For a 1.5 GW site:

- **Transmission voltage**: 230kV or 345kV incoming
- **Substations required**: 2-3 dedicated substations (N+1 redundancy)
- **Transformers**: 10-15 step-down transformers (150-200 MVA each)
- **Distribution voltage**: 13.8kV or 34.5kV to buildings
- **Switchgear**: Medium-voltage switchgear for distribution
- **Capital cost**: $150M - $250M for electrical infrastructure alone

**Power Redundancy Configurations**

| Configuration | Description | Availability | Cost Premium |
|---------------|-------------|--------------|--------------|
| N | Single feed, no redundancy | 99.0% - 99.5% | Baseline |
| N+1 | Redundant components, single path | 99.7% - 99.9% | +15% - 25% |
| 2N | Fully redundant dual path | 99.95% - 99.99% | +50% - 80% |
| 2(N+1) | Dual path with component redundancy | 99.995%+ | +80% - 120% |

**Recommendation**: N+1 for utility feeds (2-3 independent transmission lines), 2N for critical GPU cluster power distribution.

#### Power Cost Considerations

Electricity costs vary significantly by region and represent a major operational expenditure:

**Regional Power Costs (Industrial Rate, 2024-2025)**

| State/Region | Average Cost ($/kWh) | Annual Cost (1 GW, 70% utilization) |
|--------------|---------------------|-------------------------------------|
| Washington State | $0.025 - $0.030 | $153M - $184M |
| Oregon | $0.028 - $0.035 | $172M - $214M |
| Iowa | $0.030 - $0.038 | $184M - $233M |
| Illinois | $0.035 - $0.042 | $214M - $257M |
| Ohio | $0.038 - $0.045 | $233M - $276M |
| Virginia | $0.040 - $0.048 | $245M - $294M |
| Georgia | $0.042 - $0.050 | $257M - $306M |
| Nevada | $0.045 - $0.055 | $276M - $337M |
| Texas | $0.040 - $0.060 | $245M - $367M* |

*Texas costs highly variable due to deregulated market; spot pricing can spike to $9/kWh during grid stress

**Financial Impact at Scale**

For 5 GW infrastructure at 70% average utilization:

- **Annual electricity consumption**: 30.66 billion kWh
- **At $0.035/kWh (Iowa)**: $1.07 billion annually
- **At $0.050/kWh (Georgia)**: $1.53 billion annually
- **10-year difference**: $4.6 billion

**Renewable Energy Considerations**

Many states offer renewable energy programs that can stabilize long-term costs:

- **Power Purchase Agreements (PPAs)**: Lock in rates for 10-20 years
- **On-site solar/wind**: Can offset 10-30% of load in favorable locations
- **Renewable Energy Certificates (RECs)**: Meet sustainability goals while using grid power
- **Battery storage**: Shift load to off-peak hours, reduce demand charges

---

### 1.2 Fiber Connectivity and WAN Access

Network connectivity is the second critical constraint for multi-datacenter training. While power determines *if* you can build, fiber determines *how well* sites can coordinate.

#### Fiber Infrastructure Requirements

**Intra-Site (Within Datacenter Campus)**

- **Bandwidth**: 400-800 Gbps between buildings
- **Technology**: Single-mode fiber, DWDM for multi-wavelength
- **Distance**: Typically <2 km between buildings
- **Latency**: <100 microseconds
- **Cost**: $50K - $150K per fiber run, need 100+ strands

**Inter-Site (Between Datacenters)**

For multi-datacenter training with 90%+ efficiency:

| Distance | Latency Target | Bandwidth Required | Technology | Monthly Cost |
|----------|----------------|--------------------|-----------|--------------
| Metro (<100 km) | <1 ms | 100-400 Gbps | Dark fiber or wavelengths | $50K - $200K |
| Regional (100-500 km) | 5-25 ms | 50-100 Gbps | Dark fiber or wavelengths | $100K - $400K |
| Long-haul (500-2500 km) | 25-100 ms | 10-50 Gbps | Wavelength services | $200K - $800K |

**Dark Fiber vs. Wavelength Services**

| Approach | Description | Advantages | Disadvantages | Cost |
|----------|-------------|------------|---------------|------|
| **Dark Fiber** | Lease unlit fiber strands | Full control, unlimited bandwidth, one-time optics cost | High upfront, limited routes | $8K-25K/mile one-time + $2K-5K/mile annually |
| **Wavelength Services** | Lease lit wavelengths (10G, 100G, 400G) | No equipment, faster deployment, predictable cost | Bandwidth caps, recurring cost | $3K-15K per 100G wavelength monthly |
| **Hybrid** | Own metro dark fiber, lease long-haul wavelengths | Optimized cost-performance | Complexity | Variable |

**Microsoft/OpenAI Example**: $10+ billion investment in fiber contracts to interconnect datacenters nationwide, likely combination of dark fiber acquisitions and long-term wavelength commitments.

#### Latency Requirements for Multi-Datacenter Training

**Training Efficiency vs. Distance**

Based on NVIDIA Nemotron-4 340B and OpenDiLoCo results:

| Distance | Latency (RTT) | Synchronization Strategy | Expected Efficiency | Example City Pairs |
|----------|--------------|--------------------------|---------------------|-------------------|
| <50 km | <0.5 ms | Synchronous (frequent) | 98-99% | San Jose - San Francisco |
| 50-150 km | 0.5-5 ms | Synchronous (moderate) | 95-98% | Ashburn - Richmond |
| 150-500 km | 5-25 ms | Hierarchical (infrequent) | 92-96% | Atlanta - Charlotte |
| 500-1500 km | 25-75 ms | DiLoCo/Async | 90-94% | Dallas - Chicago |
| 1500-3000 km | 75-150 ms | DiLoCo/Async | 88-92% | New York - Los Angeles |

**Key Insight from NVIDIA Nemotron-4**: Achieved 96% efficiency at ~1,000 km (~20ms latency) using partial-data parallel optimizer with communication chunking and overlap.

**Recommendation**:
- **Primary sites**: <500 km apart (regional) for 95%+ efficiency with synchronous training
- **Backup/expansion sites**: 500-1500 km acceptable with DiLoCo (90%+ efficiency)
- **Avoid**: >2,000 km between primary training sites unless using fully asynchronous methods

#### Fiber Route Assessment

**Critical Evaluation Criteria**

When assessing fiber routes between potential sites:

1. **Route Diversity**
   - Minimum 2 physically diverse paths (different conduits/poles)
   - Protects against construction cuts, natural disasters
   - Example: Meta's datacenter pairs have 3-4 diverse fiber routes

2. **Carrier Availability**
   - Presence of Tier 1 carriers (AT&T, Lumen, Zayo, Verizon)
   - Competitive pricing requires 3+ carrier options
   - Dark fiber availability (Zayo, Lightower, regional providers)

3. **Latency Validation**
   - Actual measured latency, not theoretical (add 20-30% for routing)
   - Speed of light in fiber: ~5 microseconds per kilometer
   - Real-world: 6-7 microseconds per kilometer due to routing

4. **Existing Hyperscaler Presence**
   - If Google/AWS/Meta already connected the cities, fiber infrastructure proven
   - Carrier-neutral meet-me facilities exist
   - Pricing competitive due to existing demand

**Fiber Availability by Region**

| Corridor | Fiber Density | Latency | Carriers | Assessment |
|----------|---------------|---------|----------|------------|
| Northern Virginia - Atlanta | Excellent | ~15ms | 10+ | Optimal |
| Chicago - Iowa/Nebraska | Good | ~8ms | 6+ | Excellent |
| Oregon - Nevada | Moderate | ~12ms | 4+ | Good |
| Ohio - Virginia | Excellent | ~18ms | 8+ | Optimal |
| Texas - Arizona | Moderate | ~20ms | 5+ | Good |
| California - Oregon | Good | ~15ms | 6+ | Excellent |

---

### 1.3 Cooling Water Access and Climate Considerations

Thermal management at gigawatt scale requires massive heat rejection—equivalent to a small power plant. Cooling strategy and water availability are site-selection critical factors.

#### Cooling Water Requirements

**Heat Rejection Calculations**

For a 1.5 GW facility at PUE 1.20:

- **IT equipment heat**: 1.25 GW (thermal)
- **Cooling system heat rejection**: 250 MW (from PUE overhead)
- **Total heat rejection**: 1.5 GW (thermal)

**Evaporative Cooling Water Consumption**

Using cooling towers (most common for large-scale datacenters):

- **Evaporation rate**: ~1.8 gallons per ton-hour
- **1.5 GW heat rejection**: ~430,000 tons of cooling
- **Water consumption**: ~774,000 gallons per hour
- **Daily consumption**: ~18.6 million gallons
- **Annual consumption**: ~6.8 billion gallons (20,800 acre-feet)

**For comparison**:
- Medium-sized city (100,000 people): ~10-15 million gallons per day
- 1.5 GW datacenter: Equivalent to ~125,000 people water consumption

**Water Source Requirements**

| Water Source | Volume Capacity | Reliability | Regulatory Complexity | Cost |
|--------------|-----------------|-------------|----------------------|------|
| Municipal water | 10-50 MGD | High | Low | High ($3-8 per 1000 gal) |
| Surface water (lake/river) | Unlimited | Medium | High | Low ($0.50-2 per 1000 gal) |
| Groundwater (wells) | 5-20 MGD | High | Medium | Medium ($1-4 per 1000 gal) |
| Reclaimed/recycled water | 5-30 MGD | Medium | Medium | Medium ($2-5 per 1000 gal) |
| Combination | Variable | Highest | High | Optimized |

**Water Cost Impact**

At 18.6 million gallons per day:

- **Municipal water at $5/1000 gal**: $93,000 daily = $33.9M annually
- **Surface water at $1/1000 gal**: $18,600 daily = $6.8M annually
- **Savings over 10 years**: $271 million

**Climate Considerations for Free Cooling**

"Free cooling" (using outside air or water) reduces energy consumption by 20-40% during favorable conditions:

**Annual Free Cooling Hours (% of year)**

| Location | Free Air Cooling Hours (<65°F) | Free Economizer Hours (<55°F) | Annual Savings Potential |
|----------|-------------------------------|------------------------------|--------------------------|
| Prineville, OR | 7,200 (82%) | 6,100 (70%) | 35-40% |
| Council Bluffs, IA | 6,500 (74%) | 5,200 (59%) | 30-35% |
| Columbus, OH | 5,800 (66%) | 4,500 (51%) | 25-30% |
| Ashburn, VA | 5,400 (62%) | 4,100 (47%) | 25-28% |
| Atlanta, GA | 4,600 (53%) | 3,200 (37%) | 20-25% |
| Phoenix, AZ | 3,100 (35%) | 1,800 (21%) | 12-18% |
| Las Vegas, NV | 3,400 (39%) | 2,100 (24%) | 15-20% |

**Key Insight**: Cooler climates (Pacific Northwest, Midwest) provide 30-40% cooling energy savings, equivalent to $50M-100M annually per 1.5 GW site.

#### Advanced Cooling Technologies

**Direct-to-Chip Liquid Cooling**

For next-generation GPUs (B100/B200 at 1,000W+), liquid cooling becomes mandatory:

| Technology | Heat Removal Capacity | Water Consumption | PUE Impact | Capital Cost |
|------------|----------------------|-------------------|------------|--------------|
| Air cooling + CRAC | Up to 700W per GPU | 100% (evaporative tower) | 1.30-1.50 | Baseline |
| Rear-door heat exchangers | Up to 900W per GPU | 80% (reduced) | 1.20-1.30 | +15-25% |
| Direct-to-chip cold plates | Up to 1,500W per GPU | 60% (primary) | 1.12-1.20 | +30-40% |
| Immersion cooling | Up to 2,000W per GPU | 40% (minimal) | 1.08-1.15 | +50-70% |

**Water Availability Risk Assessment**

Critical for long-term site viability:

**High Risk (Avoid)**
- Western US (Arizona, Nevada, Southern California): Severe multi-decade drought
- Texas (specific regions): Declining groundwater, competing demand
- High regulatory scrutiny on new large water users

**Medium Risk (Requires robust permitting)**
- Georgia, North Carolina: Growing metro areas, competing demand
- Virginia (some counties): Groundwater limitations in certain areas
- Illinois: Groundwater permits can be complex

**Low Risk (Preferred)**
- Oregon, Washington: Abundant surface water, rainfall
- Iowa, Nebraska: Strong groundwater availability
- Ohio: Great Lakes proximity, abundant water
- Minnesota, Wisconsin: Abundant surface water

**Meta's Approach**: Sites in Prineville, OR (Deschutes River), Los Lunas, NM (Rio Grande water rights), Forest City, NC (reliable rainfall). All secured multi-decade water rights before construction.

---

### 1.4 Regulatory Environment and Incentives

State and local regulatory environments can accelerate or derail multi-billion dollar datacenter projects. Understanding regulatory complexity, permitting timelines, and available incentives is critical for site selection.

#### State Tax Incentive Comparison

**Sales Tax Exemptions on Equipment**

| State | Server/Equipment Sales Tax Exemption | Electricity Sales Tax | Estimated 10-Year Savings (1.5 GW site) |
|-------|-------------------------------------|----------------------|------------------------------------------|
| Oregon | 100% exempt | 100% exempt | $450M - $600M |
| Iowa | 100% exempt | 100% exempt | $420M - $550M |
| Virginia | 100% exempt (qualified datacenters) | 50% exempt | $350M - $480M |
| Ohio | 100% exempt (qualified datacenters) | 100% exempt | $400M - $530M |
| Nevada | 100% exempt (qualified datacenters) | 100% exempt | $380M - $500M |
| Georgia | 100% exempt (qualified datacenters) | 100% exempt | $390M - $510M |
| North Carolina | 80% exempt | 100% exempt | $340M - $450M |
| Illinois | Varies by county | Varies | $200M - $350M |
| Texas | Varies by county/district | Varies | $150M - $300M |

**Additional State Incentives**

**Virginia**
- Sales tax exemption: 100% on equipment and electricity (must meet $150M investment threshold)
- Property tax: Negotiable abatements (often 50-100% for 10-20 years)
- Job creation incentives: $5,000-8,000 per job
- **Total estimated savings**: $350M - $500M per 1.5 GW site over 10 years

**Iowa**
- Sales tax exemption: 100% on equipment, electricity, cooling
- Property tax: Varies by county, typically 50-100% abatement for 10+ years
- Investment tax credit: Up to $15M per project
- **Total estimated savings**: $420M - $550M per 1.5 GW site over 10 years

**Ohio**
- Sales tax exemption: 100% on qualified equipment
- Property tax: 75-100% abatement for 10-15 years (negotiable)
- Job creation tax credits: Varies
- **Total estimated savings**: $400M - $530M per 1.5 GW site over 10 years

**Oregon**
- No sales tax (statewide)
- Property tax: Enterprise Zone exemptions (75-100% for 15 years)
- Energy tax credits: Available for renewable energy/efficiency
- **Total estimated savings**: $450M - $600M per 1.5 GW site over 10 years

**Key Insight**: State incentives can offset $200M-600M in costs per site, equivalent to 5-15% of total site capital expenditure.

#### Permitting Complexity and Timeline

**Typical Permitting Requirements**

| Permit Type | Issuing Authority | Typical Timeline | Complexity | Failure Risk |
|-------------|-------------------|------------------|------------|--------------|
| Environmental Impact Assessment (EIA) | State EPA | 6-12 months | High | Medium |
| Water withdrawal permit | State/County | 3-9 months | Medium-High | Medium |
| Air quality permit (generators) | State EPA | 3-6 months | Medium | Low |
| Building permits | County/City | 2-4 months | Low | Low |
| Electrical interconnection | Utility + State PUC | 12-24 months | High | Medium-High |
| Zoning approval/variance | County/City | 2-6 months | Medium | Medium |
| Wetlands/endangered species | Federal (USACE, FWS) | 6-18 months | High | Medium-High |

**Total Permitting Timeline: 18-36 months** (many processes can run in parallel)

**Permitting Risk Factors**

**High Risk (Require extensive mitigation)**
- Sites near wetlands or endangered species habitat
- Water withdrawal from stressed watersheds
- Locations with active environmental opposition
- Areas with restrictive zoning (residential proximity)

**Medium Risk (Manageable with proper planning)**
- Industrial-zoned areas with some environmental sensitivity
- Moderate groundwater availability
- Established datacenter regions with precedent

**Low Risk (Streamlined permitting)**
- Pre-approved datacenter parks (Virginia, Iowa)
- Industrial brownfield sites
- Locations with existing utility infrastructure
- States with datacenter-friendly legislation

**Case Study: Virginia's Datacenter Alley (Loudoun County)**

- **Streamlined permitting**: County has dedicated datacenter review process
- **Pre-approved zoning**: Datacenter overlay districts
- **Established precedent**: 70%+ of US internet traffic passes through
- **Utility preparedness**: Dominion Energy has pre-built capacity
- **Average time to permit**: 12-18 months (vs. 24-36 months elsewhere)

---

### 1.5 Land Availability and Construction Timelines

#### Land Requirements

**Site Footprint Calculations**

For a 1.5 GW facility supporting 200,000-250,000 GPUs:

| Component | Area Required | Notes |
|-----------|---------------|-------|
| Datacenter buildings (4-6 buildings) | 1.5M - 2.0M sq ft | 250-350K sq ft per building |
| Power substations (2-3 stations) | 50,000 - 100,000 sq ft | Includes transformers, switchgear |
| Cooling infrastructure | 200,000 - 400,000 sq ft | Cooling towers, chillers, pumps |
| Emergency generators | 50,000 - 100,000 sq ft | N+1 generator farms |
| Network/fiber PoP buildings | 20,000 - 40,000 sq ft | Carrier meet-me rooms |
| Parking and admin buildings | 100,000 - 200,000 sq ft | Operations staff, visitors |
| Roads, security perimeter | 200,000 - 400,000 sq ft | Internal roads, checkpoints |
| Stormwater management | 150,000 - 300,000 sq ft | Retention ponds, drainage |
| Future expansion (Phase 2) | 1.0M - 1.5M sq ft | Land banking for growth |
| **Total Land Required** | **150-250 acres** | Allows for 50% future expansion |

**Real-World Examples:**
- Meta Prineville, OR: 345 acres (2.6 million sq ft across 5 buildings)
- Microsoft Des Moines, IA: 260 acres
- Google Council Bluffs, IA: 1,300 acres (includes massive future expansion)
- Amazon Ashburn, VA: Clusters of 30-100 acre sites

**Land Acquisition Strategy**

| Approach | Description | Advantages | Disadvantages | Typical Cost |
|----------|-------------|------------|---------------|--------------|
| **Greenfield Purchase** | Buy undeveloped land | Full control, optimized layout | Longer permitting, infrastructure build-out | $10K-100K per acre |
| **Brownfield Redevelopment** | Former industrial site | Existing utilities, expedited permits | Remediation costs, layout constraints | $50K-200K per acre |
| **Datacenter Park** | Pre-permitted datacenter zone | Fast permitting, utility-ready | Higher cost, less customization | $150K-400K per acre |
| **Build-to-Suit Lease** | Developer builds, long-term lease | No upfront land cost, faster deployment | Higher long-term cost, less control | $0 upfront, $15-30/sq ft annually |

**Recommendation**: Combination approach
- **Site 1 (Primary)**: Greenfield purchase in established datacenter region (Virginia, Iowa) for cost optimization and control
- **Site 2 (Primary)**: Datacenter park or brownfield for speed to deployment
- **Site 3 (Backup/Expansion)**: Land option or build-to-suit lease for flexibility

#### Construction Timeline

**Critical Path Analysis for 1.5 GW Datacenter**

**Pre-Construction Phase (12-18 months)**
- Month 0-3: Site selection, due diligence, land acquisition
- Month 3-9: Permitting (EIA, water, building, electrical)
- Month 6-12: Utility coordination (substation design, transmission)
- Month 9-18: Design and engineering (architecture, MEP, civil)
- Month 12-18: General contractor selection, bidding

**Construction Phase (18-24 months)**
- Month 0-6: Site preparation (grading, utilities, roads)
- Month 0-12: Electrical infrastructure (substations, distribution)
- Month 3-18: Building shell construction (4-6 buildings, staggered)
- Month 9-20: MEP installation (cooling, power distribution, fire suppression)
- Month 12-22: IT infrastructure (raised floors, cable trays, racks)
- Month 18-24: Testing and commissioning

**Equipment Installation Phase (6-12 months)**
- Month 0-3: Rack installation (1,000-2,000 racks per month)
- Month 2-8: Network equipment installation (switches, cables)
- Month 4-10: Server/GPU installation (phased by pod)
- Month 6-12: Testing, burn-in, acceptance

**Total Timeline: 36-54 months from site selection to full production**

**Acceleration Strategies**

To reduce timeline to 30-36 months:

1. **Modular Construction**
   - Pre-fabricated datacenter modules (Baselayer, Microsoft's ITPAC)
   - Reduces construction time by 30-40%
   - Trade-off: Less customization, higher per-sq-ft cost

2. **Fast-Track Permitting**
   - Select sites in pre-approved datacenter zones
   - Engage permitting consultants early
   - Parallel permit applications where possible

3. **Early Utility Engagement**
   - Begin utility discussions 24-36 months in advance
   - Fund utility infrastructure improvements
   - Lock in capacity commitments early

4. **Phased Deployment**
   - Build 1-2 buildings first (30-50% capacity)
   - Expand based on demand and learning
   - Reduces upfront capital, accelerates first production date

**Meta's Deployment Timeline (Observed)**
- Prineville Data Center 1: Site announced 2010, operational 2011 (15 months)
- Subsequent buildings: 12-18 months each
- Key enabler: Close partnership with utility (PacifiCorp), pre-permitted land

**Google's Deployment Timeline**
- Council Bluffs: Site announced 2007, first building operational 2009 (24 months)
- Multi-phase expansion: New building every 18-24 months
- Key enabler: Massive land acquisition (1,300 acres) enables continuous expansion

---

### 1.6 Labor Market and Talent Availability

Operating a multi-gigawatt AI training infrastructure requires 150-300 highly skilled personnel per site. Labor availability, cost, and quality are critical site selection factors.

#### Required Personnel by Function

**Per 1.5 GW Site (200,000-250,000 GPUs)**

| Function | Headcount | Salary Range | Annual Cost |
|----------|-----------|--------------|-------------|
| **Datacenter Operations** | | | |
| Site Director / GM | 1 | $250K - $350K | $300K |
| Datacenter Operations Managers | 4 | $150K - $200K | $700K |
| Electrical Engineers (24/7 coverage) | 12 | $100K - $140K | $1.5M |
| HVAC/Mechanical Engineers | 8 | $90K - $130K | $960K |
| Datacenter Technicians (24/7) | 40 | $60K - $90K | $3.0M |
| Security Personnel (24/7) | 24 | $50K - $70K | $1.44M |
| **IT Infrastructure** | | | |
| Infrastructure Director | 1 | $250K - $350K | $300K |
| Network Engineers (Senior) | 12 | $140K - $200K | $2.0M |
| Storage Engineers | 6 | $120K - $170K | $900K |
| Linux Systems Administrators | 16 | $100K - $150K | $2.0M |
| **ML/AI Operations** | | | |
| ML Infrastructure Director | 1 | $300K - $450K | $375K |
| ML Platform Engineers | 20 | $180K - $280K | $4.6M |
| Distributed Systems Engineers | 16 | $180K - $280K | $3.7M |
| DevOps/SRE Engineers | 12 | $150K - $220K | $2.2M |
| **Support Functions** | | | |
| Procurement/Supply Chain | 4 | $80K - $120K | $400K |
| Finance/Admin | 4 | $70K - $110K | $360K |
| EHS/Sustainability | 2 | $90K - $130K | $220K |
| **Total per Site** | **183** | | **$24.9M** |

**Total for 3 Sites: 550 personnel, $75M annual labor cost**

#### Regional Talent Market Assessment

**Tier 1: Abundant AI/ML Talent (Optimal)**

| Metro Area | AI/ML Engineers Available | Avg. Salary Premium | Datacenter Presence | Assessment |
|------------|---------------------------|---------------------|---------------------|------------|
| Seattle, WA | 25,000+ | +40% | Excellent (AWS, MSFT) | Optimal but expensive |
| San Francisco Bay Area | 40,000+ | +60% | Excellent (Google, Meta) | Best talent, highest cost |
| Austin, TX | 8,000+ | +20% | Good (Tesla, Oracle) | Good balance |
| Atlanta, GA | 6,000+ | +10% | Excellent (Google, Microsoft) | Excellent value |

**Tier 2: Emerging AI/ML Hubs (Good)**

| Metro Area | AI/ML Engineers Available | Avg. Salary Premium | Datacenter Presence | Assessment |
|------------|---------------------------|---------------------|---------------------|------------|
| Northern Virginia (DC metro) | 12,000+ | +25% | Excellent (AWS dominance) | Optimal for East Coast |
| Columbus, OH | 3,000+ | +5% | Good (Google, Meta presence) | Excellent value |
| Raleigh-Durham, NC | 4,000+ | +10% | Good (Google, Apple) | Good balance |
| Phoenix, AZ | 3,000+ | +15% | Moderate | Emerging |
| Des Moines/Iowa City, IA | 800+ | -10% | Good (Google, Microsoft, Meta) | Talent limited but cost-effective |

**Tier 3: Datacenter Regions with Limited AI Talent (Requires Build-Out)**

| Metro Area | AI/ML Engineers Available | Avg. Salary Premium | Datacenter Presence | Assessment |
|------------|---------------------------|---------------------|---------------------|------------|
| Prineville, OR | <100 (recruit from Portland) | Base | Excellent (Meta, Apple) | Remote work + rotation model |
| Council Bluffs, IA | <50 (recruit from Omaha) | -5% | Excellent (Google) | Google proven model works |

**Talent Acquisition Strategies**

**For Tier 1/2 Markets (Abundant Talent)**
- Competitive but not excessive compensation
- Leverage local university partnerships (Georgia Tech, Ohio State, UT Austin)
- Attract from existing hyperscaler operations (AWS, Google, Microsoft)
- Retention focus: Challenging work, cutting-edge technology

**For Tier 3 Markets (Talent Scarce)**
- **Remote work hybrid model**: Core team on-site, specialized roles remote
- **Rotation model**: Engineers spend 1-2 weeks on-site, 2-3 weeks remote
- **Compensation premium**: +15-25% to attract talent to smaller metros
- **Housing assistance**: Relocation support, temporary housing
- **Training programs**: Hire local operations/datacenter staff, upskill to ML/AI roles

**Case Study: Google Council Bluffs (Tier 3 Market)**
- 500+ employees in metro area of 62,000 people
- Strategy: Recruit from Omaha (1 hour away), Des Moines (2 hours), national talent willing to relocate
- Retention: Strong compensation, relocation benefits, excellent campus facilities
- Success: Operational since 2009, multiple expansions, low attrition

**University Partnerships for Talent Pipeline**

| State | Key Universities | Relevant Programs | Annual Graduates (CS/EE) |
|-------|------------------|-------------------|--------------------------|
| Virginia | UVA, Virginia Tech, George Mason | CS, ECE, Data Science | 3,000+ |
| Ohio | Ohio State, Case Western, Cincinnati | CS, ECE, AI/ML | 2,500+ |
| Georgia | Georgia Tech, Emory | CS, ECE, AI/ML | 2,000+ |
| Iowa | Iowa, Iowa State | CS, ECE | 800+ |
| Oregon | Oregon, Oregon State | CS, ECE | 1,000+ |
| Nevada | UNLV, UNR | CS, ECE | 600+ |
| Illinois | UIUC, Northwestern | CS, ECE, AI/ML | 3,500+ |

**Recommendation**:
- **Site 1 (Primary HQ)**: Atlanta, Columbus, or Northern Virginia for talent depth and datacenter precedent
- **Site 2**: Iowa or Ohio for cost efficiency and proven hyperscaler model
- **Site 3**: Oregon or secondary Virginia location for geographic diversity

---

## Section 2: Multi-State Strategy

### 2.1 Why 2-3 States Required

Single-state concentration creates unacceptable risk and operational constraints for 5GW deployment.

#### Power Delivery Limits

**Constraint**: No single utility will provision 5 GW to one location.

As analyzed in Section 1.1:
- Typical maximum single-site capacity: **1.5-2.5 GW**
- 5 GW requirement necessitates: **Minimum 2 sites, preferably 3**

**Real-World Validation**:
- Google: Multiple GW-scale sites in Iowa, Oklahoma, Oregon, Virginia
- Meta: Distributed across Oregon (Prineville), Iowa (Altoona), Georgia (Newton), New Mexico, North Carolina
- Microsoft: Sites in Virginia, Arizona, Iowa, Wyoming, others

#### Regulatory and Tax Risk Concentration

**Risk**: Single-state concentration exposes project to policy changes.

**Scenarios that have occurred**:

1. **Tax Policy Changes**
   - State eliminates datacenter tax exemptions (seen in some states post-2020)
   - Impact: $200M-400M additional tax burden over 10 years

2. **Environmental Restrictions**
   - Water withdrawal restrictions during drought
   - Renewable energy mandates (100% renewable by year X)
   - Impact: Operational constraints or forced infrastructure changes

3. **Regulatory Scrutiny**
   - State-level data residency requirements
   - Noise ordinance changes (residential expansion near datacenter)
   - Impact: Limiting expansion or forced operational changes

**Mitigation via Multi-State Strategy**:
- Diversify regulatory exposure across 2-3 states
- If one state becomes restrictive, shift growth to others
- Leverage interstate competition for favorable treatment

**Case Study: Meta's Multi-State Approach**
- Sites in OR, IA, NM, NC, GA, VA
- Hedges against state policy risk
- Can negotiate with states using credible alternative locations

#### Disaster Recovery and Business Continuity

**Risk**: Natural disasters, power grid failures, or regional outages.

**Disaster Scenarios**:

| Disaster Type | Affected Regions | Probability (20-year) | Multi-Site Mitigation |
|---------------|------------------|----------------------|------------------------|
| Hurricane | Southeast, East Coast | High (40-60%) | Site in Midwest or West |
| Tornado | Midwest, South | Medium (20-40%) | Site spacing >500 km |
| Earthquake | West Coast | Medium (20-30% major) | Site in Midwest or East |
| Ice storm / Grid failure | Northeast, Midwest | Medium (30-50%) | N+1 redundancy, geographically dispersed |
| Wildfire | West Coast | High (50-70% impact to some areas) | Site in non-wildfire region |
| Drought (water restrictions) | Southwest, California | High (60-80%) | Site in water-abundant region |

**Multi-Datacenter Training Resilience**

With DiLoCo or hierarchical training:
- **Single datacenter failure**: Training continues at 60-70% capacity
- **Recovery time**: <24 hours to redistribute workload
- **Data loss**: Minimized with cross-site checkpointing

**Contrast with Single-Site**:
- Complete outage halts training
- Recovery time: Days to weeks (depending on damage)
- Potential data loss: Last checkpoint interval

**Recommendation**:
- **Minimum 2 sites in different grid regions** (SERC, MISO, WECC)
- **Preferred: 3 sites in different climate zones** (one each: coastal, midwest, mountain/west)

#### Latency Optimization for Global User Base

**Consideration**: If trained model serves global users, multi-region deployment reduces inference latency.

While not primary driver for training site selection, co-locating training with inference infrastructure provides:
- Faster model deployment (no cross-country transfer of TB-scale models)
- Unified operations team
- Shared infrastructure (network, power, cooling)

**Global Inference Latency**

| User Location | Site Location | Latency | User Experience |
|---------------|---------------|---------|-----------------|
| East Coast US | Northern Virginia | 5-20ms | Excellent |
| East Coast US | Iowa | 30-50ms | Good |
| East Coast US | Oregon | 70-90ms | Acceptable |
| West Coast US | Oregon | 5-20ms | Excellent |
| West Coast US | Iowa | 40-60ms | Good |
| Europe | Northern Virginia | 80-100ms | Acceptable |
| Asia | Oregon | 100-150ms | Moderate |

**If inference is primary use case**: Site selection should weight proximity to user population centers.

---

### 2.2 Geographic Distribution Recommendations

#### Eastern US Options

**Northern Virginia (Loudoun/Prince William Counties)**

**Advantages:**
- Dominant datacenter market (70%+ of US internet traffic)
- Excellent fiber connectivity (domestic and international subsea cables)
- Utility capacity: 1.0-1.5 GW per site (Dominion Energy)
- Streamlined permitting (dedicated datacenter review process)
- Strong tax incentives ($350M-500M over 10 years)
- Abundant AI/ML talent (12,000+ engineers in DC metro)
- Established supplier ecosystem

**Disadvantages:**
- Power cost: $0.040-0.048/kWh (moderate)
- Climate: Limited free cooling hours (62%, $180M in cooling savings foregone vs. Oregon)
- Land cost: $200K-400K per acre in datacenter parks
- Competition for resources (every hyperscaler present)
- Water: Moderate availability (requires groundwater or Potomac allocation)

**Recommended Use**: Primary East Coast site, ideal for latency to East Coast/European users

**Estimated Capacity**: 1.5 GW max per individual site

---

**Georgia (Metro Atlanta, Newton/Covington)**

**Advantages:**
- Lower power cost: $0.042-0.050/kWh
- Strong tax incentives ($390M-510M over 10 years)
- Growing AI/ML talent hub (6,000+ engineers, Georgia Tech pipeline)
- Utility capacity: 1.0-1.5 GW (Georgia Power)
- Good fiber connectivity (Atlanta is major carrier hub)
- Meta, Google, Microsoft all present

**Disadvantages:**
- Climate: Limited free cooling (53%, $210M in cooling savings foregone vs. Oregon)
- Hurricane risk (though inland sites less vulnerable)
- Water availability: Moderate (competing demand from Atlanta metro growth)
- Permitting: 24-30 months typical

**Recommended Use**: Secondary East Coast site, or primary for cost optimization

**Estimated Capacity**: 1.5 GW max per site

---

**Ohio (Columbus, New Albany)**

**Advantages:**
- Excellent power cost: $0.038-0.045/kWh
- Strong tax incentives ($400M-530M over 10 years)
- Good free cooling hours (66%, $150M cooling savings vs. Georgia)
- Abundant water (Great Lakes proximity)
- Central US location (excellent latency to both coasts)
- Google, Meta, AWS present
- Strong university talent pipeline (Ohio State, Case Western)

**Disadvantages:**
- Moderate AI/ML talent pool (3,000+ but growing)
- Climate: Cold winters (HVAC complexity, but benefit for cooling)
- Fiber: Good but not Virginia-tier for international connectivity
- Tornado risk (moderate)

**Recommended Use**: Optimal balance of cost, performance, and risk. Excellent choice for primary or secondary site.

**Estimated Capacity**: 1.5-2.0 GW per site

---

#### Midwest Options

**Iowa (Des Moines, Council Bluffs, Altoona)**

**Advantages:**
- **Best-in-class tax incentives**: $420M-550M savings over 10 years
- Low power cost: $0.030-0.038/kWh ($460M savings vs. Virginia over 10 years at 1 GW)
- Excellent free cooling hours (74%, $190M cooling savings vs. Georgia)
- Abundant groundwater and surface water
- Proven hyperscaler success (Google, Meta, Microsoft all operational)
- Utility capacity: 1.0-1.5 GW (MidAmerican Energy)
- Low land cost: $15K-40K per acre

**Disadvantages:**
- Limited AI/ML talent pool (800+ engineers, must recruit nationally)
- Tornado risk (moderate, but datacenters designed for resilience)
- Climate extremes (cold winters, hot summers)
- Remote locations (Council Bluffs metro: 62,000 people)

**Recommended Use**: Optimal for cost-sensitive deployment. Proven model for operations in smaller metro areas.

**Estimated Capacity**: 1.5 GW per site

**Google Council Bluffs Success**: Operational since 2009, multiple expansions, demonstrates viability

---

**Illinois (Chicago metro area)**

**Advantages:**
- Major metro area (9.6M people)
- Excellent fiber connectivity (major carrier hub)
- Abundant talent pool (UIUC, Northwestern pipelines)
- Good utility capacity: 1.0-1.5 GW (ComEd, Exelon)
- Central US location

**Disadvantages:**
- Higher power cost: $0.035-0.042/kWh
- Variable tax incentives (county-dependent, $200M-350M over 10 years)
- Complex permitting (varies by county)
- High land costs near metro
- Limited free cooling hours compared to Iowa

**Recommended Use**: Good for talent-intensive operations, less optimal for pure cost efficiency

**Estimated Capacity**: 1.0-1.5 GW per site

---

#### Western US Options

**Oregon (Prineville, The Dalles, Hillsboro)**

**Advantages:**
- **Best-in-class cooling**: 82% free cooling hours ($240M cooling savings vs. Virginia over 10 years)
- No sales tax (statewide), excellent tax incentives ($450M-600M over 10 years)
- Low power cost: $0.028-0.035/kWh (hydroelectric from Columbia River)
- Abundant water (surface water rights)
- Proven success: Meta Prineville, Apple Prineville, Google The Dalles
- Renewable energy: 70%+ hydro/wind

**Disadvantages:**
- Limited utility capacity: 500 MW - 1.0 GW per site (PacifiCorp)
- Remote locations (Prineville: 10,000 people)
- Very limited local talent (must recruit from Portland or nationally)
- Earthquake risk (moderate, Cascadia subduction zone)
- Wildfire risk (moderate and increasing)

**Recommended Use**: Excellent for Phase 1 deployment or sustainability-focused projects. Limited capacity constrains mega-scale deployment.

**Estimated Capacity**: 500 MW - 1.0 GW per site (max 2 sites in region before exhausting capacity)

---

**Nevada (Reno, Las Vegas area)**

**Advantages:**
- Strong tax incentives ($380M-500M over 10 years)
- Business-friendly regulatory environment
- Growing fiber connectivity (Reno positioned between Bay Area and Salt Lake City)
- Google, Apple, Switch operational
- Utility capacity: 800 MW - 1.2 GW (NV Energy)

**Disadvantages:**
- Limited free cooling hours (39%, hot desert climate)
- Water scarcity (severe drought, regulatory risk)
- Moderate-high power cost: $0.045-0.055/kWh
- Limited talent pool (must recruit nationally)
- Wildfire smoke impacts air quality (data center air filtration challenge)

**Recommended Use**: Secondary or backup site. Water scarcity is growing concern for long-term sustainability.

**Estimated Capacity**: 800 MW - 1.2 GW per site

---

### 2.3 Latency Considerations Between Sites

#### Target Latency Thresholds

Based on NVIDIA Nemotron-4 340B (96% efficiency at 1,000 km) and OpenDiLoCo results:

**Synchronous Training (Frequent Synchronization)**

| Latency (RTT) | Distance | Training Efficiency | Recommended Synchronization |
|---------------|----------|---------------------|----------------------------|
| <1 ms | <50 km | 98-99% | Every microbatch |
| 1-5 ms | 50-200 km | 95-98% | Every batch |
| 5-25 ms | 200-500 km | 92-96% | Every few batches |
| 25-50 ms | 500-1000 km | 90-94% | Hierarchical (infrequent) |

**Asynchronous Training (DiLoCo-style)**

| Latency (RTT) | Distance | Training Efficiency | Recommended Synchronization |
|---------------|----------|---------------------|----------------------------|
| 50-100 ms | 1000-2000 km | 88-92% | Every 100-500 batches |
| 100-200 ms | 2000-4000 km | 85-90% | Every 500-1000 batches |

#### City Pair Latency Matrix

**Major Datacenter City Pairs (Measured Latency)**

| Origin | Destination | Distance (km) | Fiber Latency (RTT) | Training Efficiency (Estimated) |
|--------|-------------|---------------|---------------------|----------------------------------|
| Northern Virginia | Columbus, OH | 480 km | 6-8 ms | 95-97% |
| Northern Virginia | Atlanta, GA | 840 km | 12-16 ms | 93-96% |
| Northern Virginia | Des Moines, IA | 1,450 km | 20-28 ms | 91-94% |
| Columbus, OH | Des Moines, IA | 1,040 km | 15-22 ms | 92-95% |
| Atlanta, GA | Des Moines, IA | 1,360 km | 19-26 ms | 91-94% |
| Des Moines, IA | Prineville, OR | 2,150 km | 30-42 ms | 88-92% (DiLoCo) |
| Atlanta, GA | Prineville, OR | 3,450 km | 48-65 ms | 86-90% (DiLoCo) |
| Northern Virginia | Prineville, OR | 3,750 km | 52-70 ms | 85-90% (DiLoCo) |
| Columbus, OH | Reno, NV | 2,900 km | 42-58 ms | 87-91% (DiLoCo) |
| Atlanta, GA | Reno, NV | 2,850 km | 40-55 ms | 87-91% (DiLoCo) |

**Recommended Site Pair Strategies**

**Strategy 1: Regional Proximity (Synchronous Training)**
- **Site Pair**: Columbus, OH ↔ Northern Virginia
- **Distance**: 480 km
- **Latency**: 6-8 ms
- **Efficiency**: 95-97%
- **Advantages**: Highest efficiency, both sites in datacenter-mature regions
- **Disadvantages**: Both East Coast (limited geographic diversity)

**Strategy 2: Balanced National (Hybrid Synchronous/Hierarchical)**
- **Site Pair**: Northern Virginia ↔ Des Moines, IA
- **Distance**: 1,450 km
- **Latency**: 20-28 ms
- **Efficiency**: 91-94%
- **Advantages**: Good balance of efficiency and geographic diversity, excellent cost savings (Iowa)
- **Disadvantages**: Moderate latency requires hierarchical synchronization

**Strategy 3: Tri-Site National (Hierarchical/DiLoCo)**
- **Site 1**: Northern Virginia (East)
- **Site 2**: Des Moines, IA (Central)
- **Site 3**: Prineville, OR (West)
- **Max Distance**: 3,750 km (VA ↔ OR)
- **Max Latency**: 52-70 ms
- **Efficiency**: 85-90% (with DiLoCo)
- **Advantages**: Optimal geographic diversity, climate diversity, disaster recovery
- **Disadvantages**: Requires DiLoCo or asynchronous training methods

#### Fiber Route Validation

**Critical Step**: Measure actual fiber latency, don't assume theoretical.

**Validation Process**:
1. **Identify fiber carriers** serving both locations (Zayo, Lumen, etc.)
2. **Request latency measurements** (carrier will provide or allow test)
3. **Add 20-30% buffer** for routing overhead (fiber isn't perfectly straight)
4. **Verify route diversity** (critical for redundancy)

**Example**: Columbus, OH to Des Moines, IA
- Straight-line distance: 805 km
- Theoretical latency (c/1.5): 5.4 ms one-way, 10.8 ms RTT
- Actual fiber route: ~1,040 km (routing via Chicago hubs)
- Measured latency: 15-22 ms RTT
- Budget for engineering: 20-25 ms RTT

---

### 2.4 State-Specific Advantages Summary

#### Decision Matrix: State-by-State Comparison

**Scoring Methodology**: 1-5 scale (5 = best, 1 = worst)

| State | Power Cost | Power Capacity | Tax Incentives | Fiber Connectivity | Free Cooling | Water Access | Talent Pool | Permitting Speed | Overall Score |
|-------|------------|----------------|----------------|--------------------|--------------|--------------| ------------|------------------|---------------|
| **Iowa** | 5 | 4 | 5 | 3 | 5 | 5 | 2 | 4 | **33** |
| **Ohio** | 5 | 5 | 5 | 4 | 4 | 5 | 3 | 4 | **35** |
| **Oregon** | 5 | 3 | 5 | 3 | 5 | 5 | 2 | 3 | **31** |
| **Virginia** | 3 | 4 | 4 | 5 | 3 | 3 | 5 | 5 | **32** |
| **Georgia** | 4 | 4 | 4 | 4 | 2 | 3 | 4 | 3 | **28** |
| **Nevada** | 2 | 3 | 4 | 3 | 1 | 1 | 2 | 4 | **20** |
| **Illinois** | 4 | 4 | 3 | 5 | 3 | 4 | 4 | 2 | **29** |

**Top-Tier Choices (Score 32+)**:
1. **Ohio** (35): Best overall balance, strong across all dimensions
2. **Iowa** (33): Best cost, proven hyperscaler success
3. **Virginia** (32): Best connectivity and talent, established market

**Recommended 3-Site Strategy Based on Scoring**:

**Configuration A: Cost-Optimized, High Performance**
- **Site 1 (Primary, 2.0 GW)**: Columbus, Ohio
- **Site 2 (Primary, 2.0 GW)**: Des Moines/Altoona, Iowa
- **Site 3 (Backup/Expansion, 1.0 GW)**: Northern Virginia
- **Total**: 5.0 GW
- **Advantages**: Lowest 10-year OpEx, all sites proven by hyperscalers, <30ms latency between primary sites
- **Trade-offs**: Limited West Coast presence (add Oregon if inference latency critical)

**Configuration B: Geographically Balanced**
- **Site 1 (Primary, 2.0 GW)**: Northern Virginia
- **Site 2 (Primary, 2.0 GW)**: Des Moines, Iowa
- **Site 3 (Backup/Expansion, 1.0 GW)**: Prineville, Oregon
- **Total**: 5.0 GW
- **Advantages**: National coverage, disaster resilience, climate diversity
- **Trade-offs**: Higher latency to Oregon (requires DiLoCo), talent recruitment challenge in Iowa/Oregon

**Configuration C: Talent-Optimized**
- **Site 1 (Primary, 2.0 GW)**: Northern Virginia
- **Site 2 (Primary, 2.0 GW)**: Metro Atlanta, Georgia
- **Site 3 (Backup/Expansion, 1.0 GW)**: Columbus, Ohio
- **Total**: 5.0 GW
- **Advantages**: All sites in major metros with strong AI/ML talent pools
- **Trade-offs**: Higher power costs, less geographic diversity (all East/Southeast)

**Executive Recommendation**: **Configuration A (Ohio + Iowa + Virginia)** provides optimal balance of cost ($200M+ annual savings vs. Configuration C), performance (low latency), risk mitigation, and proven hyperscaler validation.

---

## Section 3: Site Architecture

### 3.1 Building Design for GPU Density

Modern GPU densities (700W-1,500W per GPU) exceed traditional datacenter design parameters. Building architecture must be purpose-built for AI/ML workloads.

#### Power Density Requirements

**Traditional Datacenter vs. AI/ML Datacenter**

| Parameter | Traditional Enterprise DC | Cloud/Hyperscale DC | AI/ML Training DC |
|-----------|--------------------------|---------------------|-------------------|
| Rack power density | 5-8 kW | 10-15 kW | 30-100 kW |
| Floor power density | 100-200 W/sq ft | 200-400 W/sq ft | 400-800 W/sq ft |
| Cooling approach | CRAC units (air) | In-row cooling | Liquid cooling mandatory |
| Power distribution | 208V 3-phase | 415V 3-phase | 415V 3-phase + DC busbar |

**GPU Rack Power Calculations**

**Configuration 1: Standard Server Rack (8x H100 GPUs)**
- GPUs: 8× H100 at 700W = 5,600W
- CPU/Memory/Motherboard: 800W
- Network equipment: 300W
- Fans and PSU losses: 500W
- **Total per rack: 7.2 kW**
- **Racks per 18MW building**: ~2,500 racks = 20,000 GPUs

**Configuration 2: High-Density Rack (8x B100/B200 at 1,000W)**
- GPUs: 8× at 1,000W = 8,000W
- CPU/Memory/Motherboard: 1,000W
- Network equipment: 400W
- Fans and PSU losses: 800W
- Liquid cooling CDU overhead: 200W
- **Total per rack: 10.4 kW**
- **Racks per 18MW building**: ~1,730 racks = 13,840 GPUs

**Configuration 3: Ultra-High-Density Rack (NVIDIA GB200 NVL72)**
- GPUs: 36× Grace-Blackwell Superchips at ~1,400W = 50,400W
- Liquid cooling infrastructure: 2,000W
- Network equipment: 800W
- **Total per rack: 53.2 kW**
- **Racks per 18MW building**: ~338 racks = 12,168 GPUs

**Key Insight**: 18MW building constraint (Alibaba HPN observation) limits GPU count per building regardless of GPU density. **Liquid cooling mandatory for B200/B100 and beyond.**

#### Building Layout and Floorplan

**Typical Large-Scale AI Datacenter Building**

**Building Specifications (250,000 sq ft, 18MW IT power)**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATACENTER BUILDING (500' × 500')                │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    WHITE SPACE (GPU RACKS)                    │   │
│  │                      180,000 sq ft                            │   │
│  │                                                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │   │
│  │  │   Pod 1     │  │   Pod 2     │  │   Pod 3     │          │   │
│  │  │ 6,000 GPUs  │  │ 6,000 GPUs  │  │ 6,000 GPUs  │          │   │
│  │  │ 750 racks   │  │ 750 racks   │  │ 750 racks   │          │   │
│  │  │ 6MW         │  │ 6MW         │  │ 6MW         │          │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘          │   │
│  │                                                               │   │
│  │  Hot Aisle Containment                                       │   │
│  │  Liquid Cooling Distribution Units (CDUs)                    │   │
│  │  Overhead Cable Trays (Network + Power)                      │   │
│  │                                                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │Electrical│  │ HVAC/    │  │ Network  │  │ Storage/ │            │
│  │Rooms     │  │ Cooling  │  │ Meet-Me  │  │ NOC      │            │
│  │          │  │ Plants   │  │ Rooms    │  │          │            │
│  │10,000 sq │  │30,000 sq │  │10,000 sq │  │5,000 sq  │            │
│  │ft        │  │ft        │  │ft        │  │ft        │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              LOADING DOCK & RECEIVING (15,000 sq ft)         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

**Space Allocation**

| Area | Square Footage | % of Total | Purpose |
|------|----------------|------------|---------|
| White space (GPU racks) | 180,000 | 72% | Compute infrastructure |
| Electrical/UPS rooms | 10,000 | 4% | Power distribution, batteries |
| HVAC/Cooling plants | 30,000 | 12% | Chillers, pumps, CDUs, cooling towers |
| Network meet-me rooms | 10,000 | 4% | Spine switches, WAN connectivity |
| Storage & NOC | 5,000 | 2% | Checkpoint storage, operations center |
| Loading dock & receiving | 15,000 | 6% | Equipment delivery, staging |
| **Total** | **250,000** | **100%** | |

**Pod-Based Architecture**

Each building divided into 3-4 independent pods:
- **Fault isolation**: Failure in one pod doesn't impact others
- **Phased deployment**: Bring pods online sequentially
- **Maintenance**: Can service one pod while others operate
- **Network topology**: Hierarchical all-reduce within pod, cross-pod at higher tier

---

### 3.2 Raised Floor vs. Slab Design

#### Raised Floor Design (Traditional)

**Configuration**:
- Elevated floor 24-36 inches above structural slab
- Underfloor plenum for cold air distribution and cable routing
- Perforated tiles in cold aisles for airflow

**Advantages**:
- Flexible power and network cable routing
- Good for retrofit/reconfiguration
- Efficient cold air distribution (historically)

**Disadvantages**:
- **Higher cost**: $150-250 per sq ft (vs. $80-120 for slab)
- **Reduced ceiling height**: May limit rack height
- **Not optimal for liquid cooling**: Underfloor space better used for other purposes
- **Limited power density**: Struggles beyond 15-20 kW per rack

**Verdict**: Becoming obsolete for AI/ML datacenters. Only suitable for <10 kW/rack air-cooled deployments.

---

#### Slab-on-Grade with Overhead Distribution (Modern AI/ML Standard)

**Configuration**:
- Equipment installed directly on reinforced concrete slab
- Power and network distribution via overhead cable trays
- Liquid cooling pipes via overhead or in-rack distribution
- Hot aisle containment with rear-door heat exchangers or direct-to-chip cooling

**Advantages**:
- **Lower cost**: $80-120 per sq ft
- **Higher power density**: Supports 30-100+ kW per rack
- **Better for liquid cooling**: Overhead pipes for supply/return
- **Easier maintenance**: All infrastructure accessible from above
- **Faster construction**: No raised floor installation

**Disadvantages**:
- Less flexible for reconfiguration (cables overhead vs. underfloor)
- Requires careful initial planning (harder to change)

**Verdict**: **Recommended for all AI/ML deployments**. Used by Google, Meta, Microsoft for hyperscale facilities.

**Alibaba HPN Reference**: 15,000 GPU deployments use slab-on-grade design with overhead distribution.

---

### 3.3 Expansion Capacity Planning

#### Phased Build-Out Strategy

**Phase 1 (Months 0-36): Initial Deployment**
- **Capacity**: 100,000-150,000 GPUs across 1-2 sites
- **Power**: 700 MW - 1.2 GW contracted (includes redundancy)
- **Buildings**: 4-6 buildings (250K sq ft each)
- **Investment**: $35-50 billion

**Phase 2 (Months 24-60): Expansion**
- **Capacity**: +150,000-200,000 GPUs
- **Power**: +1.0-1.5 GW
- **Buildings**: +4-6 buildings
- **Investment**: +$35-50 billion
- **Overlap with Phase 1**: Begin Phase 2 construction while Phase 1 still deploying

**Phase 3 (Months 48-72): Full Build-Out**
- **Total capacity**: 350,000-500,000 GPUs
- **Total power**: 3.5-5.0 GW
- **Total buildings**: 12-18 buildings across 3 sites
- **Total investment**: $95-133 billion

**Land Banking Strategy**

Purchase 2-3× the land required for Phase 1:
- **Phase 1 needs**: 60-80 acres per site
- **Purchase**: 150-250 acres per site
- **Allows**: Phase 2 & 3 expansion without new site selection/permitting
- **Cost impact**: Minimal (land is small fraction of total cost)

**Real-World Example: Google Council Bluffs**
- Purchased: 1,300 acres initially
- Phase 1 (2009): 1 building
- Continuous expansion: Now 4+ buildings, multiple expansions
- Still has land for 5-10+ more buildings

#### Electrical Infrastructure Expansion

**Modular Substation Design**

Design electrical infrastructure for phased scaling:

**Phase 1 Electrical (1.0 GW site)**
- Install: 2× substations (500 MW each, N+1 redundancy)
- Transformers: 8× 100 MVA units
- Reserve space: For +2 substations in Phase 2/3

**Phase 2 Electrical (+500 MW)**
- Add: 1× substation (500 MW)
- Transformers: +4× 100 MVA units
- Activate reserved distribution capacity

**Phase 3 Electrical (+500 MW to reach 2.0 GW)**
- Add: 1× substation (500 MW)
- Complete build-out to 2.0 GW

**Design Principle**: "Build for today, design for tomorrow"
- Install transmission lines sized for full build-out (Day 1)
- Install substations/transformers incrementally (as needed)
- Avoids costly utility re-work

---

### 3.4 Physical Security Requirements

Multi-billion dollar GPU infrastructure requires enterprise-grade physical security.

#### Security Perimeter Layers

**Layer 1: Site Perimeter (Outer)**
- **Fencing**: 8-10 ft chain-link with barbed wire or anti-climb design
- **Cameras**: Every 100-150 ft, 24/7 recording
- **Lighting**: Perimeter illumination (motion-activated or constant)
- **Intrusion detection**: Buried sensors, microwave/radar systems
- **Access control**: Single vehicle entry point, staffed checkpoint

**Layer 2: Building Perimeter**
- **Access control**: Badge readers, biometric (fingerprint/facial recognition)
- **Mantrap**: Secure entry vestibules (one door closes before next opens)
- **Cameras**: All entry points, loading docks
- **Security operations center (SOC)**: 24/7 monitoring

**Layer 3: Internal Zones (White Space)**
- **Caged areas**: High-value equipment (if multi-tenant, not typical for single-org)
- **Cameras**: In white space (every aisle or critical areas)
- **Access logging**: Track all personnel entry/exit

**Layer 4: Logical Security (Physical-Digital Integration)**
- **Badge + two-factor**: Physical badge + PIN or biometric
- **Visitor management**: Pre-registration, escort requirements
- **Asset tracking**: RFID tags on removable equipment

#### Security Operations

**Staffing (per 1.5 GW site)**
- **Security officers**: 24× personnel (8 per shift, 3 shifts)
- **SOC operators**: 6× personnel (2 per shift, 3 shifts)
- **Security manager**: 1
- **Annual cost**: $1.44M (officers) + $600K (SOC/manager) = ~$2M per site

**Real-World Standards**
- **Tier III datacenter** (Uptime Institute): Armed guards, vehicle barriers, extensive CCTV
- **Government/classified**: Additional requirements (SCIF standards if handling classified AI work)

**NVIDIA GPU Theft Prevention**
- H100 GPUs: $25,000-30,000 each, easily portable
- **Countermeasures**:
  - Serial number tracking (all GPUs registered)
  - Cage lockdown (GPU racks in secured cages)
  - Escort requirements (non-employees must be escorted in white space)
  - Exit inspections (random checks at loading dock)

---

## Section 4: Regulatory and Compliance

### 4.1 Environmental Permits and Impact Assessments

#### Environmental Impact Assessment (EIA) Process

**Trigger Thresholds (State-Dependent)**

Most states require EIA for projects meeting criteria such as:
- **Power consumption**: >50 MW
- **Water consumption**: >5 million gallons per day
- **Land disturbance**: >50 acres
- **Investment**: >$100 million

**A 1.5 GW datacenter exceeds all thresholds** → Full EIA required

**EIA Timeline and Process**

| Phase | Duration | Activities | Deliverables |
|-------|----------|------------|--------------|
| Scoping | 1-2 months | Identify issues, stakeholders | Scoping document |
| Baseline studies | 4-8 months | Environmental surveys (wildlife, water, air, noise) | Baseline report |
| Impact analysis | 3-6 months | Model impacts, identify mitigation | Draft EIA |
| Public comment | 2-4 months | Public hearings, comment period | Revised EIA |
| Agency review | 2-6 months | State EPA/DNR review and approval | Final EIA, permits |
| **Total** | **12-26 months** | | |

**Key Environmental Concerns**

1. **Air Quality**
   - **Issue**: Backup diesel generators (emissions during testing/use)
   - **Mitigation**: Limit run hours, use Tier 4 generators, explore fuel cells
   - **Permit**: State air quality permit required

2. **Water Quality**
   - **Issue**: Discharge from cooling systems (temperature, chemicals)
   - **Mitigation**: Closed-loop systems, water treatment before discharge
   - **Permit**: NPDES permit (National Pollutant Discharge Elimination System)

3. **Wildlife and Habitat**
   - **Issue**: Construction impacts on local species
   - **Mitigation**: Habitat surveys, avoid critical breeding seasons, offsets
   - **Permit**: Endangered Species Act (ESA) review if threatened species present

4. **Stormwater**
   - **Issue**: Runoff from paved areas (parking, roads)
   - **Mitigation**: Retention ponds, green infrastructure
   - **Permit**: Stormwater management plan

**Cost of EIA**: $500K - $2M for studies, permitting, mitigation measures

**Risk Mitigation**:
- **Select pre-approved industrial sites**: Former factories, industrial parks (easier permitting)
- **Avoid environmentally sensitive areas**: Wetlands, endangered species habitat, floodplains
- **Early stakeholder engagement**: Meet with regulators and community before formal application

---

### 4.2 Water Usage Permits

#### Water Withdrawal Permits

**Permit Requirements by Water Source**

| Water Source | Permitting Authority | Typical Timeline | Approval Difficulty |
|--------------|---------------------|------------------|---------------------|
| Municipal water | Local utility | 2-4 months | Easy (if capacity available) |
| Groundwater wells | State DNR/Water Resources | 6-12 months | Moderate (depends on aquifer capacity) |
| Surface water (river/lake) | State DNR + Federal (navigable waters) | 9-18 months | High (competing uses, environmental impact) |
| Reclaimed water | Local wastewater utility | 3-6 months | Moderate (depends on availability) |

**Typical Permit Conditions**

1. **Withdrawal Limits**
   - Daily maximum (e.g., 20 million gallons per day)
   - Annual maximum
   - Seasonal restrictions (may limit withdrawal during drought)

2. **Monitoring and Reporting**
   - Install flow meters
   - Monthly/quarterly reporting to state agency
   - Real-time monitoring (in some states)

3. **Conservation Requirements**
   - Demonstrate efficiency measures (e.g., target PUE)
   - Water recycling/reuse plans
   - Drought contingency plan

4. **Environmental Flow Requirements** (for surface water)
   - Maintain minimum river flow downstream
   - May prohibit withdrawal during low-flow periods

**Case Study: Meta Prineville, Oregon**
- **Source**: Deschutes River (surface water)
- **Permit**: Secured water rights for cooling
- **Mitigation**: Significant investment in river conservation projects
- **Outcome**: Successful permitting, but required extensive stakeholder engagement

**Water Scarcity Risk States**

**High Risk (Difficult/Risky Permitting)**:
- Arizona, Nevada, New Mexico (severe multi-decade drought)
- Southern California (overdrafted aquifers)
- Texas (specific regions with declining groundwater)

**Recommendation**: Avoid water-scarce regions for new 1.5 GW sites. If unavoidable, pursue reclaimed water or dry cooling (expensive).

---

### 4.3 Noise Ordinances

#### Noise Sources in Datacenters

**Primary Noise Generators**

| Source | Typical Noise Level (dBA at 50 ft) | Mitigation |
|--------|-------------------------------------|------------|
| Cooling towers (evaporative) | 60-75 dBA | Acoustic louvers, distance setbacks |
| Backup generators (testing) | 75-85 dBA | Soundproof enclosures, limit test hours |
| Air-cooled chillers | 65-75 dBA | Acoustic barriers, low-noise fans |
| HVAC fans | 55-70 dBA | Acoustic treatment, variable speed drives |

**For comparison**:
- Normal conversation: 60 dBA
- Busy traffic: 70-80 dBA
- Leaf blower: 85-90 dBA

**Local Noise Ordinances (Typical Limits)**

| Zoning | Daytime Limit (7am-10pm) | Nighttime Limit (10pm-7am) |
|--------|--------------------------|----------------------------|
| Residential | 55-60 dBA | 45-50 dBA |
| Commercial | 60-65 dBA | 55-60 dBA |
| Industrial | 70-75 dBA | 65-70 dBA |

**Challenge**: Datacenters operate 24/7, nighttime limits more restrictive.

**Mitigation Strategies**

1. **Site Selection**
   - **Prefer industrial-zoned areas** (more lenient noise limits)
   - **Distance from residential** (>1,000 ft recommended)
   - **Natural barriers** (trees, hills for acoustic buffering)

2. **Equipment Design**
   - **Enclosed cooling towers** (vs. open design)
   - **Low-noise fans** (variable speed, acoustic optimization)
   - **Generator enclosures** (soundproof buildings for generators)

3. **Testing Schedules**
   - **Limit generator testing** to daytime hours
   - **Notify neighbors** in advance of testing

4. **Acoustic Modeling**
   - **Pre-construction modeling**: Predict noise levels at property line
   - **Adjustments**: Ensure compliance before construction

**Cost Impact**: Noise mitigation adds 2-5% to building costs (acoustic barriers, enclosures, low-noise equipment upgrades).

**Real-World Issue**: Meta's Henrico County, VA datacenter faced community opposition due to noise concerns. Required extensive sound studies and mitigation commitments.

---

### 4.4 Energy Efficiency Mandates

#### State and Federal Requirements

**State-Level Datacenter Energy Efficiency Laws**

Several states have enacted or proposed datacenter-specific efficiency mandates:

| State | Requirement | Effective Date | Penalties |
|-------|-------------|----------------|-----------|
| California | PUE <1.2 for new datacenters >1 MW | Proposed (not yet law) | TBD |
| Washington | Energy efficiency reporting | 2024 | None (reporting only) |
| Virginia | Optional green energy incentives | Ongoing | Voluntary |
| Oregon | Renewable energy portfolio | Ongoing | Utility-level (not datacenter-specific) |

**Federal Energy Efficiency Programs**

- **ENERGY STAR for Data Centers**: Voluntary certification
- **EPA Green Power Partnership**: Renewable energy procurement commitment
- **No mandatory federal PUE limits** (as of 2024-2025)

**Recommended Targets to Future-Proof**

Even without mandates, target aggressive efficiency:

| Metric | Target | Industry Baseline | Financial Impact (1.5 GW site, 10 years) |
|--------|--------|-------------------|-------------------------------------------|
| PUE | <1.20 | 1.50 | $370M savings |
| Renewable energy % | 50-80% | 0-30% | Variable (depends on PPA pricing) |
| Water usage | <15M gal/day | 20M gal/day | $50-100M savings |

**Renewable Energy Procurement Strategies**

1. **Power Purchase Agreements (PPAs)**
   - Contract directly with solar/wind farm
   - Lock in pricing for 10-20 years
   - Often competitive with grid pricing in Iowa, Texas, Oregon

2. **Utility Green Tariffs**
   - Many utilities offer renewable energy programs
   - Slightly higher cost (+$0.005-0.015/kWh) but guaranteed renewable

3. **On-Site Generation**
   - Solar panels on datacenter roofs/land
   - Can offset 5-15% of load (depends on site)
   - Capital intensive but long-term savings

4. **Renewable Energy Certificates (RECs)**
   - Purchase RECs to "offset" grid usage
   - Least expensive option (~$1-5 per MWh)
   - Environmental credit only (no actual green power delivery)

**Case Study: Google's 24/7 Carbon-Free Energy Goal**
- Target: Match energy usage with carbon-free energy every hour (not just annual offset)
- Strategy: Combination of wind, solar, battery storage, advanced PPAs
- Progress: Achieved 64% globally (2023), aiming for 100% by 2030

**Recommendation**: Target **50% renewable energy minimum** to hedge against future mandates and support corporate sustainability goals.

---

### 4.5 Data Sovereignty and Regulatory Compliance

#### Data Residency Requirements

**Current Landscape (2024-2025)**

**United States**:
- **No federal data residency requirements** for commercial AI training
- **Sector-specific**: Healthcare (HIPAA), financial (GLBA) have some restrictions
- **State privacy laws** (California CCPA, Virginia VCDPA) focus on user rights, not data location

**If training on international data or serving international users**:

| Jurisdiction | Requirement | Impact on Multi-DC Training |
|--------------|-------------|----------------------------|
| **European Union (GDPR)** | EU citizen data must be processable under GDPR | Can train in US with Standard Contractual Clauses (SCCs) |
| **China** | Strict data localization for Chinese citizen data | Cannot train on Chinese data outside China (requires China DC) |
| **Russia** | Personal data of Russian citizens must be stored in Russia | Cannot train on Russian data outside Russia |
| **India (proposed)** | Data localization under discussion | May require India DC if finalized |

**Risk Mitigation for International Data**

1. **Data Segregation**
   - **European data**: Train in EU-compliant US datacenter with SCCs
   - **Chinese data**: Separate training in China (if applicable)
   - **US data**: No restrictions

2. **Anonymization and Aggregation**
   - Remove PII before training
   - GDPR permits processing of anonymized data
   - Reduces regulatory burden

3. **Multi-Region Deployment** (if serving global users)
   - **Training**: Can centralize in US
   - **Inference**: Deploy regionally (EU, Asia, US)
   - Ensures low latency and regulatory compliance for inference

**Export Controls (ITAR/EAR)**

**Issue**: Advanced AI systems may be subject to export controls.

**Current Status (2024-2025)**:
- **NVIDIA H100/A100**: Export restrictions to China, Russia (requires licenses)
- **AI model weights**: Not currently export-controlled (but evolving)
- **Training infrastructure**: Generally not restricted within US

**Compliance Recommendations**:
1. **Ensure all GPUs are properly licensed** (NVIDIA/AMD handle for US customers)
2. **Monitor regulatory changes**: AI export controls are rapidly evolving
3. **Legal review**: If training for defense/government, ITAR may apply

**Multi-State Strategy for Regulatory Hedging**

**Benefit**: Regulatory risk diversification

- **Example Scenario**: California enacts strict datacenter energy limits
- **Mitigation**: Shift growth to Iowa or Ohio sites
- **Outcome**: Business continuity maintained

**Recommendation**:
- **Avoid single-state concentration** for regulatory risk mitigation
- **Monitor state legislative activity** on datacenter regulations
- **Maintain flexibility** to shift workloads between sites

---

## Section 5: Real-World Examples

### 5.1 Meta's Datacenter Locations

**Overview**: Meta operates one of the world's largest datacenter networks, purpose-built for AI/ML workloads including LLama model training.

**Key Sites**

| Location | State | Announced | Operational | Estimated Capacity | Key Features |
|----------|-------|-----------|-------------|--------------------| -------------|
| Prineville | Oregon | 2010 | 2011 | 500+ MW (5 buildings) | First custom datacenter, hydroelectric power, excellent cooling climate |
| Forest City | North Carolina | 2011 | 2012 | 300-400 MW | East Coast presence, fiber connectivity to VA |
| Altoona | Iowa | 2013 | 2014 | 400-500 MW | Low power cost, wind energy, central US location |
| Fort Worth | Texas | 2015 | 2017 | 300-400 MW | South-central US, growing |
| Los Lunas | New Mexico | 2016 | 2019 | 300-400 MW | High desert, solar energy potential |
| Henrico County | Virginia | 2017 | 2019 | 400-500 MW (expanding) | East Coast, excellent fiber, proximity to Ashburn |
| Newton | Georgia | 2018 | 2020 | 400-500 MW | Southeast US, Georgia Power partnership |
| Huntsville | Alabama | 2020 | 2022 | TBD | Tennessee Valley Authority power |
| Kuna | Idaho | 2024 | 2026 (planned) | TBD (potentially 1+ GW) | Major expansion, Pacific Northwest |

**Multi-State Strategy Observations**

1. **Geographic Diversity**: 8 states spanning all US regions
2. **Power Cost Optimization**: Heavy presence in low-cost states (OR, IA, NM)
3. **Renewable Energy**: Sites chosen for wind/solar/hydro access
4. **Phased Expansion**: Each site has multiple buildings, multi-year build-out
5. **RoCEv2 Network**: Interconnects sites with RDMA over Ethernet

**Meta's Site Selection Criteria (Inferred)**

- **Power**: Availability of 300-500 MW+ per site
- **Renewables**: Access to renewable energy (wind, solar, hydro)
- **Fiber**: Carrier-neutral connectivity
- **Incentives**: Strong state/local tax incentives
- **Talent**: Proximity to mid-size metros (not necessarily large cities)

**Key Takeaway**: Meta's distributed strategy across 8+ states validates multi-state deployment necessity for hyperscale AI infrastructure.

---

### 5.2 Google's Multi-State AI Infrastructure

**Overview**: Google pioneered large-scale datacenter development for AI/ML (TPU training since 2015). Multi-state presence supports both training and global inference.

**Key AI-Optimized Sites**

| Location | State | Operational Since | Estimated Capacity | Key Features |
|----------|-------|-------------------|--------------------| -------------|
| Council Bluffs | Iowa | 2009 | 1+ GW (1,300 acre campus) | First major hyperscale DC in Iowa, massive expansion capacity |
| Mayes County (Pryor) | Oklahoma | 2011 | 500-800 MW | Low power cost, central US, Grand River water access |
| The Dalles | Oregon | 2006 | 400-600 MW | Hydroelectric power, Columbia River cooling |
| Jackson County | South Carolina | 2013 | 300-400 MW | Southeast presence |
| New Albany | Ohio | 2019 | 400-600 MW (expanding) | East Coast alternative to Virginia, excellent fiber to VA/NC |
| Papillion | Nebraska | 2019 | TBD (expanding) | Complement to Iowa, low cost power |
| Midlothian | Texas | 2018 | 300-400 MW | South-central US |

**Multi-Datacenter Training Evidence**

- **Research**: Google DeepMind published DiLoCo (2023-2024), demonstrating multi-continent training
- **Infrastructure**: High-bandwidth fiber between Ohio and Iowa/Nebraska regions
- **Capacity**: Multi-gigawatt capability supports massive model training
- **Technology**: TPU v5, custom interconnects optimized for distributed training

**Site Selection Pattern**

Google prioritizes:
1. **Low power cost**: Heavy investment in Midwest (IA, OK, NE)
2. **Renewable energy**: Often co-located with wind farms (Iowa) or hydro (OR, WA)
3. **Massive land acquisition**: Council Bluffs (1,300 acres) enables decades of expansion
4. **Metro-area proximity**: Near mid-size metros (Omaha, Columbus) for talent

**Ohio Strategy (New Albany)**

- **First building**: 2019 (400-600 MW)
- **Expansion announced**: Additional buildings approved
- **Strategic importance**:
  - <20ms to Northern Virginia (low latency for East Coast services)
  - Lower cost than Virginia
  - Excellent fiber connectivity (Columbus is carrier hub)
  - Strong talent pool (Ohio State University)

**Key Takeaway**: Google's Ohio + Iowa regional cluster demonstrates optimal latency-cost balance for multi-site training.

---

### 5.3 Microsoft/OpenAI Site Selection

**Overview**: Microsoft operates 200+ datacenters globally. Azure AI infrastructure supports OpenAI model training (GPT-4, etc.).

**Major US AI-Optimized Sites**

| Location | State | Estimated Capacity | Key Features |
|----------|-------| -------------------|--------------|
| Quincy | Washington | 500+ MW | Hydroelectric power, low cost |
| Des Moines | Iowa | 400-600 MW | Low power cost, central US |
| Goodyear | Arizona | 500+ MW | South-west presence, expanding |
| Boydton | Virginia | 400-600 MW | East Coast, excellent fiber |
| San Antonio | Texas | 300-500 MW | South-central, renewable energy |
| Cheyenne | Wyoming | 400-600 MW | Low cost power, wind energy |
| West Des Moines | Iowa | Expanding | Complement to Des Moines campus |

**Microsoft/OpenAI Multi-Datacenter Training Initiative**

**Evidence from SemiAnalysis and industry reports**:

1. **$10+ billion fiber investment**: Long-haul contracts to interconnect datacenters nationwide
2. **GPU CRIU adoption**: Process migration capability (NVIDIA cuda-checkpoint available since 2024)
3. **Training infrastructure**: First to target multi-GW computing for single models
4. **Azure AI platform**: Supports multi-region training and inference

**Inferred Strategy**:

- **Primary training sites**: Arizona, Iowa (low cost power, large capacity)
- **Secondary sites**: Virginia, Washington (inference, latency to users)
- **Interconnect**: Dedicated fiber between sites
- **Synchronization**: Likely asynchronous parameter servers or DiLoCo-inspired methods

**Key Differences from Meta/Google**:

- **Cloud business model**: Microsoft must support customer workloads globally
- **Multi-use infrastructure**: AI training shares sites with Azure cloud services
- **Rapid scaling**: Aggressive expansion to meet OpenAI demands

**Key Takeaway**: Microsoft's $10B+ fiber investment validates the necessity and feasibility of multi-datacenter training for trillion-parameter models.

---

## Section 6: Site Selection Decision Framework

### 6.1 Weighted Scoring Model

**Methodology for Selecting 3 Sites from Candidate Locations**

**Criteria and Weights** (Total: 100 points)

| Criteria | Weight | Rationale |
|----------|--------|-----------|
| Power capacity and reliability | 20 | Primary constraint, non-negotiable |
| Power cost (10-year OpEx) | 15 | Massive financial impact ($200M+ difference) |
| Tax incentives | 10 | Can offset $200-600M over 10 years |
| Fiber connectivity | 10 | Critical for multi-DC training efficiency |
| Latency to other sites | 10 | Determines training efficiency (90-98% range) |
| Cooling climate (free cooling hours) | 8 | $100-250M impact over 10 years |
| Water availability | 8 | Long-term sustainability, regulatory risk |
| Talent availability | 7 | Operational capability, harder to quantify |
| Permitting timeline | 6 | Time-to-market, competitive advantage |
| Disaster risk (climate, seismic) | 6 | Business continuity |

**Total: 100 points**

---

### 6.2 Site Comparison Scorecard

**Scoring: 1-5 scale for each criterion (5 = best)**

| Site | Power Cap | Power Cost | Tax Incent | Fiber | Latency* | Cooling | Water | Talent | Permitting | Disaster Risk | **Total** |
|------|-----------|------------|------------|-------|----------|---------|-------|--------|------------|---------------|-----------|
| **Columbus, OH** | 5 (20) | 5 (15) | 5 (10) | 4 (8) | 5 (10) | 4 (6.4) | 5 (8) | 3 (4.2) | 4 (4.8) | 4 (4.8) | **91.2** |
| **Des Moines, IA** | 4 (16) | 5 (15) | 5 (10) | 3 (6) | 4 (8) | 5 (8) | 5 (8) | 2 (2.8) | 4 (4.8) | 3 (3.6) | **82.2** |
| **N. Virginia** | 4 (16) | 3 (9) | 4 (8) | 5 (10) | 5 (10) | 3 (4.8) | 3 (4.8) | 5 (7) | 5 (6) | 3 (3.6) | **79.2** |
| **Atlanta, GA** | 4 (16) | 4 (12) | 4 (8) | 4 (8) | 3 (6) | 2 (3.2) | 3 (4.8) | 4 (5.6) | 3 (3.6) | 3 (3.6) | **70.8** |
| **Prineville, OR** | 3 (12) | 5 (15) | 5 (10) | 3 (6) | 2 (4) | 5 (8) | 5 (8) | 2 (2.8) | 3 (3.6) | 2 (2.4) | **71.8** |
| **Reno, NV** | 3 (12) | 2 (6) | 4 (8) | 3 (6) | 2 (4) | 1 (1.6) | 1 (1.6) | 2 (2.8) | 4 (4.8) | 3 (3.6) | **50.4** |

*Latency scored relative to other candidate sites in portfolio (lower average latency = higher score)

**Top 3 Sites Based on Scoring**:

1. **Columbus, Ohio**: 91.2 points (highest overall)
2. **Des Moines, Iowa**: 82.2 points (best cost, efficiency)
3. **Northern Virginia**: 79.2 points (best connectivity, talent)

---

### 6.3 Final Site Selection Recommendation

**Recommended 3-Site Configuration**

**Site 1 (Headquarters & Primary Training): Columbus, Ohio**
- **Capacity**: 2.0 GW (250,000 GPUs at full build-out)
- **Rationale**: Best overall score, optimal balance across all criteria
- **Advantages**:
  - Excellent power cost and capacity
  - Strong tax incentives
  - Good cooling efficiency
  - Abundant water
  - Mid-size metro with growing AI talent
  - Central US location (low latency to both coasts)
- **Timeline**: 30-36 months to first production

**Site 2 (Cost-Optimized Training): Des Moines/Altoona, Iowa**
- **Capacity**: 2.0 GW (250,000 GPUs at full build-out)
- **Rationale**: Lowest total cost of ownership, proven hyperscaler model
- **Advantages**:
  - Best power cost ($460M/decade savings vs. Virginia)
  - Best tax incentives ($420M-550M over 10 years)
  - Best cooling efficiency ($190M savings vs. Georgia)
  - Proven success (Google, Meta, Microsoft operational)
- **Trade-offs**: Limited talent pool (mitigated by remote work + rotation model)
- **Timeline**: 30-36 months to first production

**Site 3 (Backup & East Coast Expansion): Northern Virginia**
- **Capacity**: 1.0 GW initial (expandable to 1.5 GW)
- **Rationale**: Best fiber connectivity, talent pool, East Coast presence
- **Advantages**:
  - Dominant datacenter market (lowest risk)
  - Best fiber connectivity (international subsea cables)
  - Largest AI/ML talent pool
  - Fastest permitting (established datacenter infrastructure)
- **Trade-offs**: Higher cost (acceptable for strategic value)
- **Timeline**: 24-30 months to first production (fastest due to streamlined permitting)

**Total Configuration**:
- **Power**: 5.0 GW contracted capacity
- **GPUs**: 500,000-700,000 (across 3 sites at full build-out)
- **Geographic diversity**: Midwest (OH, IA), East Coast (VA)
- **Latency**: <30ms between any two sites
- **Disaster resilience**: Different grid regions (MISO, PJM), climate zones

**Deployment Sequence**:

**Phase 1 (Months 0-36)**: Northern Virginia + Columbus, Ohio
- VA first (fastest permitting): 500 MW
- OH second (larger capacity): 1.0 GW
- **Total Phase 1**: 1.5 GW, 200,000-250,000 GPUs

**Phase 2 (Months 24-54)**: Iowa + VA/OH expansion
- Iowa first site: 1.0 GW
- VA expansion: +500 MW
- OH expansion: +500 MW
- **Total after Phase 2**: 3.5 GW, 400,000-500,000 GPUs

**Phase 3 (Months 48-72)**: Final build-out
- Iowa expansion: +1.0 GW
- OH expansion: +500 MW
- **Total after Phase 3**: 5.0 GW, 600,000-700,000 GPUs

**Financial Impact of Recommended Configuration**

| Item | Ohio | Iowa | Virginia | Total |
|------|------|------|----------|-------|
| Power (10 yrs, 2.0/2.0/1.0 GW @ 70% util) | $4.66B @ $0.040/kWh | $4.37B @ $0.034/kWh | $2.45B @ $0.042/kWh | $11.48B |
| Tax incentives (10 yrs) | -$400M | -$420M | -$350M | -$1.17B |
| Cooling savings (vs. baseline GA) | -$150M | -$190M | $0 (baseline) | -$340M |
| **Net 10-Year OpEx Advantage vs. All-Georgia Config** | | | | **-$1.51B** |

**Our recommended configuration saves $1.5 billion over 10 years vs. concentrating all capacity in Georgia, while providing superior disaster resilience and regulatory diversification.**

---

## Conclusion

Site selection for multi-gigawatt AI training infrastructure is a multi-billion dollar strategic decision with 10-20 year consequences. The analysis in this chapter demonstrates:

1. **Multi-state deployment is mandatory**, driven by power delivery limits (1.5-2.5 GW per site maximum), regulatory risk diversification, and disaster recovery requirements.

2. **Power availability is the primary constraint**, followed by fiber connectivity, water access, and regulatory environment. Climate (cooling efficiency) and tax incentives can drive $200M-500M in savings per site over 10 years.

3. **Optimal 3-site configuration balances cost, performance, and risk**:
   - **Columbus, Ohio** (headquarters, best overall score)
   - **Des Moines, Iowa** (cost optimization, proven model)
   - **Northern Virginia** (connectivity, talent, backup)

4. **Latency between sites must be <50ms for synchronous training** (95%+ efficiency) or <100ms for hierarchical/DiLoCo training (90%+ efficiency). Recommended configuration achieves <30ms maximum latency.

5. **Regulatory complexity is manageable** with proper planning. Environmental permits, water rights, and noise ordinances require 18-36 month lead time but are navigable in datacenter-friendly states.

6. **Real-world validation from Meta, Google, and Microsoft** confirms the feasibility and necessity of multi-state strategies. All three hyperscalers operate 6-8+ sites spanning multiple states.

The site selection framework and scoring methodology provided in this chapter enable executive teams to make data-driven decisions optimizing for total cost of ownership, operational performance, and strategic resilience.

---

**Chapter 3 Complete**

**Next Chapter**: Network Architecture and Topology Design

---

## References

1. Meta Investor Relations, "Data Center Locations and Investments" (2020-2024)
2. Google Cloud Locations, public disclosures (2006-2024)
3. Microsoft Azure Global Infrastructure documentation (2024)
4. NVIDIA Technical Blog, "Nemotron-4 340B Multi-Datacenter Training" (2024)
5. Alibaba HPN, "A Data Center Network for Large Language Model Training", ACM SIGCOMM 2024
6. SemiAnalysis, "Multi-Datacenter Training Infrastructure Analysis" (2024)
7. Uptime Institute, "Data Center Site Infrastructure Tier Standard: Topology" (2024)
8. US Energy Information Administration, "Electric Power Monthly" (2024)
9. State economic development agencies (VA, OH, IA, OR, GA, NV), datacenter incentive programs
10. US Environmental Protection Agency, NPDES Permit Program
11. National Oceanic and Atmospheric Administration (NOAA), climate data
12. American Society of Heating, Refrigerating and Air-Conditioning Engineers (ASHRAE), "Thermal Guidelines for Data Processing Environments" (2021)
