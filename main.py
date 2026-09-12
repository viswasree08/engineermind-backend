from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import math

app = FastAPI(
    title="EngineerMind AI - Physics & Diagnostics Engine",
    description="Deterministic calculation and rule verification engine for external CFD",
    version="1.0.0"
)

# Enable CORS so your future frontend can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input data contract: user provides 4 core physical parameters
class CaseInput(BaseModel):
    velocity: float              # Freestream velocity (m/s)
    chord_length: float          # Characteristic length / chord (meters)
    fluid_density: float = 1.225 # Air at standard sea level (kg/m^3)
    viscosity: float = 1.789e-5  # Dynamic viscosity of air (Pa.s)
    target_y_plus: float = 1.0   # Boundary layer resolution target

# Output data contract: deterministic physics returned to HUD
class PhysicsOutput(BaseModel):
    reynolds_number: float
    mach_number: float
    flow_regime: str
    boundary_layer_thickness_mm: float
    first_cell_height_mm: float
    inlet_distance_m: float
    outlet_distance_m: float
    lateral_distance_m: float
    recommended_turbulence_model: str
    recommended_coupling: str
    recommended_spatial_discretization: str
    senior_engineer_rationale: str

@app.get("/")
def health_check():
    return {"status": "EngineerMind AI Physics Engine is active"}

@app.post("/calculate-physics", response_model=PhysicsOutput)
def calculate_physics(data: CaseInput):
    # Safety guardrails: avoid negative or zero physical inputs
    if data.velocity <= 0 or data.chord_length <= 0:
        raise HTTPException(
            status_code=400, 
            detail="Velocity and chord length must be strictly positive numbers."
        )

    # 1. Calculate Reynolds Number: Re = (rho * v * L) / mu
    re = (data.fluid_density * data.velocity * data.chord_length) / data.viscosity

    # 2. Calculate Mach Number: Ma = v / speed_of_sound (340.3 m/s at 288.15 K)
    speed_of_sound = 340.3
    mach = data.velocity / speed_of_sound

    # Determine flow regime
    if mach >= 0.3:
        regime = "Compressible Subsonic (Density variations must be modeled)"
    elif re < 5e5:
        regime = "Incompressible Laminar / Transitional"
    else:
        regime = "Incompressible Fully Turbulent"

    # 3. Boundary Layer Thickness (Flat Plate Estimate): delta = 0.37 * c / (Re^0.2)
    delta_meters = (0.37 * data.chord_length) / (re ** 0.2)
    delta_mm = delta_meters * 1000.0

    # 4. First Cell Height Calculation (delta_s for target y+)
    # Skin friction coefficient: Cf = 0.0583 * Re^(-0.2)
    cf = 0.0583 * (re ** -0.2)
    # Wall shear stress: tau_w = 0.5 * rho * v^2 * Cf
    tau_w = 0.5 * data.fluid_density * (data.velocity ** 2) * cf
    # Friction velocity: u_tau = sqrt(tau_w / rho)
    u_tau = math.sqrt(tau_w / data.fluid_density)
    # First cell height: delta_s = (y_plus * mu) / (rho * u_tau)
    delta_s_meters = (data.target_y_plus * data.viscosity) / (data.fluid_density * u_tau)
    delta_s_mm = delta_s_meters * 1000.0

    # 5. Domain Dimensions (15c upstream, 25c wake, 15c farfield)
    inlet = data.chord_length * 15.0
    outlet = data.chord_length * 25.0
    lateral = data.chord_length * 15.0

    # 6. Prescribed Model & Senior Engineer Rationale
    rec_model = "k-omega SST (Menter)"
    rec_coupling = "Coupled (Pressure-Velocity)"
    rec_schemes = "Second-Order Upwind (Momentum & Turbulent Quantities)"
    rationale = (
        "SST k-omega is recommended because external aerodynamics involves boundary layers under "
        "adverse pressure gradients. Standard k-epsilon overpredicts eddy viscosity and delays predicted "
        "separation. Target y+ <= 1 directly resolves the viscous sublayer without artificial wall functions."
    )

    return PhysicsOutput(
        reynolds_number=round(re, 2),
        mach_number=round(mach, 3),
        flow_regime=regime,
        boundary_layer_thickness_mm=round(delta_mm, 2),
        first_cell_height_mm=round(delta_s_mm, 5),
        inlet_distance_m=round(inlet, 2),
        outlet_distance_m=round(outlet, 2),
        lateral_distance_m=round(lateral, 2),
        recommended_turbulence_model=rec_model,
        recommended_coupling=rec_coupling,
        recommended_spatial_discretization=rec_schemes,
        senior_engineer_rationale=rationale
    )
