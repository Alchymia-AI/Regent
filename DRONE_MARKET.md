# Regent for Drone and Autonomous UAV Systems

## Why drones specifically

A drone running a multi-hour survey mission has three hard constraints that eliminate most language models as a cognitive layer.

**Memory wall.** A transformer accumulates a KV cache that grows with every token processed over the mission. A 4-hour mission at modest token rates hits gigabytes. Embedded hardware does not have gigabytes to spare. Regent's SSM state is fixed at ~2 MB for the 7B model regardless of session length. The drone thinks for 4 hours and uses the same memory it used in the first second.

**Offline operation.** Connectivity is intermittent or nonexistent. API-based models fail when the link drops. Regent runs locally on the drone's edge compute. There is no cloud dependency.

**Actionable grounding.** A drone receiving a fabricated navigation instruction executes it physically. The verification head produces a per-decision grounding score before the action is committed. A planner can reject or escalate any decision that scores below threshold. That is a safety primitive, not a post-hoc filter.

---

## How Regent maps onto a drone system

| Regent component | Role on the drone |
|---|---|
| Language backbone | Mission reasoning, natural-language instruction parsing, status narration |
| EPG encoder | Stores waypoints, no-fly zones, obstacles, terrain features, mission objectives as typed nodes injected natively — not as text in the prompt |
| Essence vector | Encodes mission state: urgency (0–10), risk tolerance, battery conservation mode, priority weighting between speed and safety |
| Verification head | Scores every navigation decision; planner holds or re-routes when confidence drops below threshold |
| SSM recurrent state | Maintains compressed mission memory across the full flight without growth |

---

## Data required to train a drone-capable Regent

Training happens in two layers: the language backbone (standard Regent phases 1–4) and drone-specific fine-tuning that teaches the model mission reasoning and navigation.

---

### Layer 1 — Language backbone data (standard pipeline)

Run the standard 4-phase pipeline first. The backbone needs general language competence before domain-specific fine-tuning.

**Phase 1 — Base language modelling**

General text: web crawl, Wikipedia, books, technical documentation. Standard Regent training. No drone-specific data required here.

**Phase 2 — Identity SFT**

Regent conversation logs and instruction-following pairs. Include drone-adjacent technical writing: FAA regulations, ICAO standards, UAV operator manuals, mission briefing formats, airspace classification documents. Sources:

- FAA UAS regulations and advisory circulars — `https://www.faa.gov/uas`
- ICAO Annex 2 and RPAS manual
- ArduPilot and PX4 documentation — `https://ardupilot.org/ardupilot` and `https://docs.px4.io`
- ROS 2 navigation documentation — `https://docs.nav2.ros.org`
- DJI developer SDK documentation
- Military UAV operation manuals where publicly available (FOIA releases, declassified)
- Search and rescue UAV operation guides (AUVSI, public SAR organizations)
- Academic papers on UAV mission planning (ArXiv cs.RO)

**Phase 3 — Verification head**

Grounded/corrupted pairs. For drones: pairs where one version of a navigation decision is grounded in the drone's actual sensor state and one is fabricated. Generate these from flight simulation logs (see simulation sources below).

**Phase 4 — Alignment (DPO)**

Preference pairs where conservative, safety-aware navigation decisions are preferred over aggressive or risky ones. Generate from human pilot ratings of simulated mission outcomes.

---

### Layer 2 — Navigation and mission fine-tuning data

This is what teaches the drone to self-direct. The model needs to learn the mapping from sensor state + mission objective → navigation decision, grounded in real and simulated flight data.

---

#### Step 1 — Simulation flight logs (start here)

Simulators produce unlimited labeled data with ground truth. Begin training on simulation before touching real hardware.

**AirSim (Microsoft)**
Open-source drone and car simulator built on Unreal Engine. Produces photorealistic sensor data: RGB camera, depth, IMU, GPS, lidar. Run with PX4 SITL (software-in-the-loop) for realistic flight dynamics.
- Repository: `https://github.com/microsoft/AirSim`
- Generates: flight telemetry, camera frames, obstacle positions, waypoint sequences, collision events
- Use: generate millions of navigation episodes across varied terrain (urban, rural, indoor, weather conditions)

**Gazebo + PX4 SITL**
The standard robotics simulator. Less photorealistic than AirSim but faster to run at scale and tightly integrated with ROS 2 and PX4 autopilot stack.
- PX4 SITL: `https://docs.px4.io/main/en/simulation/gazebo_classic.html`
- Generates: full flight controller state, sensor fusion output, mission logs, fail-safe events
- Use: generate structured navigation sequences with known-good and known-bad outcomes for verification head training

**FlightGoggles (MIT)**
High-fidelity visual simulator for agile flight. Designed for learning-based navigation research. Supports collision-free and collision trajectories.
- Repository: `https://github.com/mit-fast/FlightGoggles`
- Use: visual navigation training, obstacle avoidance in tight spaces

**RotorS (ETH Zurich)**
Gazebo-based MAV simulator with well-validated aerodynamic models for multi-rotor platforms.
- Repository: `https://github.com/ethz-asl/rotors_simulator`

**What to extract from simulation logs**

For each simulated flight episode, extract:
- State sequence: position (x, y, z), velocity, heading, battery level, wind estimate, obstacle distances
- Decision sequence: waypoint targets, speed commands, altitude changes, hold/reroute events
- Outcome: mission success, collision, low-battery abort, out-of-bounds
- Grounding label: decisions made with accurate sensor state (grounded=1) vs. decisions made under sensor noise or spoofed state (grounded=0)

Format each episode as a sequence of (state, decision, outcome, grounding_score) tuples. The model learns to predict the next decision given state context and to score its confidence via the verification head.

---

#### Step 2 — Real flight datasets

Use publicly available datasets to ground the model in real sensor noise, real aerodynamics, and real environment variation that simulators do not fully replicate.

**EuRoC MAV Dataset**
Micro aerial vehicle flights in indoor and outdoor environments. Stereo camera + IMU. Ground truth from motion capture and laser tracker. 11 sequences varying difficulty.
- Source: ETH Zurich ASL — `https://rpg.ifi.uzh.ch/docs/IJRR17_Burri.pdf`
- Data: `https://rpg.ifi.uzh.ch/data/MAV_datasets`
- Use: visual-inertial odometry, state estimation training

**TartanAir Dataset**
Simulation-based but highly diverse: 30+ environment types (urban, forest, ocean, snow, night, rain), 1000+ trajectories, stereo/depth/IMU/lidar/optical flow. Designed specifically for learning-based navigation.
- Source: Carnegie Mellon — `https://theairlab.org/tartanair-dataset`
- Use: multi-environment generalisation, domain randomisation

**UZH-FPV Drone Racing Dataset**
High-speed first-person-view flights. IMU at 1000 Hz, event camera, standard camera. Tests extreme agility and fast obstacle response.
- Source: University of Zurich — `https://fpv.ifi.uzh.ch`
- Use: high-speed navigation, aggressive manoeuvre learning

**KITTI Odometry Benchmark**
Autonomous driving dataset but widely used for visual odometry transfer to aerial. Lidar + stereo + GPS.
- Source: Karlsruhe Institute of Technology — `https://www.cvlibs.net/datasets/kitti/eval_odometry.php`
- Use: lidar-based localisation, loop closure

**DARPA Subterranean (SubT) Dataset**
Multi-robot exploration in underground environments (caves, tunnels, urban underground). Includes aerial vehicles. High difficulty, GPS-denied.
- Source: `https://subtchallenge.world/open-datasets`
- Use: GPS-denied navigation, map building under uncertainty

**BlackBird Dataset (MIT)**
Aggressive indoor flight with photorealistic rendering. 9 vehicles, 168 flight trajectories, IMU + camera.
- Source: MIT CSAIL — `https://github.com/mit-fast/blackbird-dataset`
- Use: indoor agile flight, fast motion blur handling

**OpenDroneMap Aerial Imagery**
Community-sourced georeferenced aerial survey imagery. Not flight dynamics but useful for terrain understanding and mapping.
- Source: `https://opendronemap.org`

---

#### Step 3 — Navigation reasoning data

The model needs to learn to reason about navigation, not just pattern-match sensor-to-action. This requires language-grounded navigation data: descriptions of environments paired with navigation decisions and outcomes.

**Vision-Language Navigation (VLN) datasets**

These pair natural language instructions with navigation trajectories. Originally designed for indoor robots but the structure transfers directly to drone mission planning.

- **R2R (Room-to-Room)** — `https://bringmeaspoon.org` — 22,000 instructions paired with navigation paths in Matterport 3D environments
- **REVERIE** — remote object grounding with navigation in large 3D environments
- **TouchDown** — navigation from natural language in real-world street-level imagery (Google Street View)
- **AerialVLN** — aerial vision-language navigation specifically for UAVs in outdoor environments — `https://github.com/AirVLN/AirVLN`

**Mission planning corpora**

Convert structured mission formats into natural language for SFT:

- MAVLink mission files (`.plan` format from QGroundControl and Mission Planner) — parse waypoint sequences and convert to natural language mission descriptions paired with the structured plan
- PX4 flight log `.ulg` files — convert telemetry to narrative: "At t=42s the drone detected obstacle at 3m range and adjusted heading 15° right"
- Public flight logs shared by the ArduPilot and PX4 communities on their forums

**Incident and failure reports**

FAA UAS incident reports teach the model what goes wrong and why. They are written in structured natural language with cause analysis — high-quality grounding data for the verification head.
- FAA Aviation Safety Hotline database — `https://asias.faa.gov`
- NTSB accident database filtered for UAS — `https://www.ntsb.gov/investigations`

---

#### Step 4 — EPG node schema for navigation

Define the EPG node categories specific to drone navigation. These are injected as prefix context on every inference call and shape the model's grounding.

| Category | Example nodes |
|---|---|
| `waypoint` | `{key: "WP-04", value: "lat 37.42, lon -122.08, alt 50m, loiter 30s"}` |
| `no_fly_zone` | `{key: "NFZ-KSJC-5mi", value: "5 nautical mile ring around San Jose Intl, class C"}` |
| `obstacle` | `{key: "tower-grid-C3", value: "antenna tower 120m AGL at grid C3, strobe active"}` |
| `terrain` | `{key: "terrain-ridge-north", value: "ridge elevation 340m, 2km north of launch"}` |
| `mission_objective` | `{key: "obj-primary", value: "thermal scan grid sector 7, 90% coverage required"}` |
| `weather` | `{key: "wind-current", value: "14kt NW, gusts 22kt, ceiling 2500ft AGL"}` |
| `asset_state` | `{key: "battery", value: "68%, estimated 18min endurance remaining"}` |
| `constraint` | `{key: "curfew", value: "operations must cease by 1900 local per permit 2026-0441"}` |

Confidence scores on these nodes reflect sensor certainty (GPS fix quality, time since last weather update, obstacle detection confidence).

---

#### Step 5 — Reinforcement learning in simulation

After supervised pre-training on the datasets above, use RL to optimise for mission outcomes that cannot be fully expressed in demonstration data.

**Environment setup**

Use AirSim or Gazebo with a reward function that penalises:
- Collision (large negative)
- No-fly zone violation (large negative)
- Battery-critical flight (moderate negative)
- Mission objective miss (moderate negative)

And rewards:
- Waypoint completion
- Mission objective coverage
- Safe return to home
- Verification head high-confidence decisions (small positive — encourages conservative paths when uncertain)

**Algorithm**

PPO (Proximal Policy Optimisation) is the standard choice. The policy network is the Regent model with frozen backbone and trainable head layers. The observation is the EPG node set (current state) plus the instruction. The action space is a discretised navigation command: heading delta, altitude delta, speed, hold.

**Curriculum**

1. Simple open-field point-to-point navigation (no obstacles)
2. Static obstacle fields
3. Dynamic obstacles (other aircraft, moving objects)
4. GPS-denied with visual-inertial only
5. Adverse weather (wind, rain — simulated)
6. Multi-objective missions with re-planning
7. Coordinated multi-drone operations

---

#### Step 6 — Human pilot demonstration data

Record experienced drone pilots completing missions in simulation. Use behavioural cloning to initialise the RL policy before fine-tuning. This is faster and more stable than learning from scratch with RL.

- Partner with commercial drone operators (survey, inspection, delivery companies)
- Record QGroundControl sessions with operator narration
- Label decisions: routine, cautious, abort, emergency
- Use the narration to generate EPG node updates in real time

---

## Deployment stack

```
Drone hardware
├── Edge compute (Jetson Orin or equivalent)
│   ├── Regent 7B INT4 (~4 GB VRAM)
│   ├── Flight controller interface (MAVLink / ROS 2 bridge)
│   └── Sensor fusion node (GPS, IMU, lidar, camera)
├── Flight controller (PX4 or ArduPilot)
└── Sensor suite (lidar, stereo camera, IMU, GPS, barometer)

Regent runtime
├── EPG nodes: mission-loaded at takeoff, updated during flight
├── Essence vector: set by ground station (urgency, risk mode)
├── Verification head: gates every navigation command
└── SSM state: persists mission context for full flight duration
```

The ground station sends updated EPG nodes and Essence settings over MAVLink. The drone's Regent instance runs inference locally. Navigation decisions are passed to the flight controller only when the verification head scores above the configured threshold.

---

## Minimum viable training run for a drone prototype

To produce a demonstration-ready drone model on the 370M or 1.5B config:

1. Run standard Regent base training on general text (phase 1) — existing pipeline
2. Fine-tune on ArduPilot/PX4 documentation + FAA regulations + VLN datasets (phase 2 SFT)
3. Generate 500K simulation episodes from AirSim + PX4 SITL across 5 terrain types
4. Train verification head on grounded vs. sensor-spoofed episode pairs (phase 3)
5. Run PPO curriculum stages 1–4 in simulation (simple navigation to GPS-denied)
6. Record 50 hours of human pilot demonstrations in simulation, add as BC warm-start

Expected outcome: a model that can parse natural-language mission briefs, maintain mission context over a multi-hour flight, reject navigation commands it scores as uncertain, and re-plan around new obstacles injected as EPG nodes.
