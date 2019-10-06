# Routines for setting up the solver for streaming problems
#
# should take care of the steps between setting up Navier-Stokes system
# and the integrator. Should accept the domain dimensions, grid spacing, time
# step size, Reynolds number,
#
using LinearAlgebra

const TU = Tuple{Nodes{Dual},Nodes{Dual},Edges{Primal},Vector{Float64}}
const TF = Tuple{VectorData,VectorData,Vector{Float64},Vector{Float64}}

setup_system(Re,Δx,xlim,ylim,Δt,coords;ddftype=Fields.Goza) =
        NavierStokes(Re,Δx,xlim,ylim,Δt,
                    X̃ = coords,
                    isstore = true,
                    isasymptotic = false,
                    isfilter = false, ddftype = ddftype)

initialize_state(sys) = (Nodes(Dual,size(sys)),Nodes(Dual,size(sys)),Edges(Primal,size(sys)),[0.0,0.0])
initialize_force(sys) = (VectorData(sys.X̃),VectorData(sys.X̃),Vector{Float64}(),Vector{Float64}())

function initialize_solver(Re,Δx,xlim,ylim,Δt,body::Body,motion::RigidBodyMotions.RigidBodyMotion;ddftype=Fields.Goza)

  coords = VectorData(body.x,body.y)

  sys = setup_system(Re,Δx,xlim,ylim,Δt,coords;ddftype=ddftype)

  w₀ = Nodes(Dual,size(sys))
  ΔX = Edges(Primal,w₀)
  f1 = VectorData(coords)

  u = initialize_state(sys)
  f = initialize_force(sys)

  plans = ((t,u) -> Fields.plan_intfact(t,u,sys),(t,u) -> Fields.plan_intfact(t,u,sys),
           (t,u) -> Identity(),(t,u)-> I)

  gradopx = Regularize(sys.X̃,cellsize(sys);I0=origin(sys),issymmetric=true,ddftype=ddftype,graddir=1)
  gradopy = Regularize(sys.X̃,cellsize(sys);I0=origin(sys),issymmetric=true,ddftype=ddftype,graddir=2)
  dEx = InterpolationMatrix(gradopx,sys.Fq,sys.Vb)
  dEy = InterpolationMatrix(gradopy,sys.Fq,sys.Vb)

  streaming_r₁(u,t) = TimeMarching.r₁(u,t,sys,motion)
  streaming_r₂(u,t) = TimeMarching.r₂(u,t,sys,motion,dEx,dEy,body.cent)
  streaming_plan_constraints(u,t::Float64) = TimeMarching.plan_constraints(u,t,sys)

  return IFHERK(u,f,sys.Δt,plans,streaming_plan_constraints,
                      (streaming_r₁,streaming_r₂),rk=TimeMarching.RK31,isstored=true), sys

end

function TimeMarching.r₁(u::TU,t::Float64,sys::NavierStokes,motion::RigidBodyMotions.RigidBodyMotion)
    _,ċ,_,_,α̇,_ = motion(t)
    return zero(u[1]), TimeMarching.r₁(u[1],t,sys), lmul!(-1,curl(sys.L\u[1])), [real(ċ),imag(ċ)]
end

function TimeMarching.r₂(u::TU,t::Float64,sys::NavierStokes,motion::RigidBodyMotions.RigidBodyMotion,dEx,dEy,centroid)
  fact = 2 # not sure how to explain this factor yet.
  _,ċ,_,_,α̇,_ = motion(t)
  U = (real(ċ),imag(ċ))
   # -ΔX̂⋅∇v₁
  Δx⁻¹ = 1/cellsize(sys)

  sys.Fq .= curl(sys.L\u[1]) # -v₁
  sys.Vb .= dEx*sys.Fq # dv₁/dx*Δx
  sys.Vb.u .*= -fact*Δx⁻¹*u[4][1]
  sys.Vb.v .*= -fact*Δx⁻¹*u[4][1]
  Vb = deepcopy(sys.Vb) # -X⋅dv₁/dx
  sys.Vb .= dEy*sys.Fq # dv₁/dy*Δx
  sys.Vb.u .*= -fact*Δx⁻¹*u[4][2]
  sys.Vb.v .*= -fact*Δx⁻¹*u[4][2]
  Vb .+= sys.Vb # -X⋅dv₁/dx - Y.⋅dv₁/dy
  return U + α̇ × (sys.X̃ - centroid), Vb, Vector{Float64}(), Vector{Float64}()

end

function TimeMarching.plan_constraints(u::TU,t::Float64,sys::NavierStokes)
    B₁ᵀ, B₂ = TimeMarching.plan_constraints(u[1],t,sys) # These are used by both the first and second equations
    return (B₁ᵀ,B₁ᵀ,f->zero(u[3]),f->zero(u[4])),
            (B₂,B₂,u->Vector{Float64}(),u->Vector{Float64}())
end
