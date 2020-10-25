# Routines for setting up the flow solver for streaming problems
#
using LinearAlgebra

import RigidBodyTools: RigidBodyMotion

import ConstrainedSystems: r₁, r₂, B₁ᵀ, B₂, plan_constraints


#=
Make the tuple data structures. The state tuple holds the first and second asymptotic
level vorticity, the Eulerian displacement field for the first level, and the component
of the unscaled centroid displacement of the body. The last two parts have no constraints,
so we set the force to empty vectors.
=#
const TU = Tuple{Nodes{Dual},Nodes{Dual},Edges{Primal},Vector{Float64}}
const TF = Tuple{VectorData,VectorData,Vector{Float64},Vector{Float64}}

setup_system(Re,Δx,xlim,ylim,Δt,coords;ddftype=CartesianGrids.Goza) =
        NavierStokes(Re,Δx,xlim,ylim,Δt,
                    X̃ = coords,
                    isstore = true,
                    isasymptotic = false,
                    isfilter = false, ddftype = ddftype)

initialize_state(sys) = Nodes(Dual,size(sys)),Nodes(Dual,size(sys)),Edges(Primal,size(sys)),[0.0,0.0]

function initialize_state(sys,nbody)
  rigid = zeros(3*nbody)
  return Nodes(Dual,size(sys)),Nodes(Dual,size(sys)),Edges(Primal,size(sys)),rigid
end

initialize_force(sys) = (VectorData(sys.X̃),VectorData(sys.X̃),Vector{Float64}(),Vector{Float64}())

function initialize_solver(Re,Δx,xlim,ylim,Δt,body::Body,motion::RigidBodyMotion;ddftype=CartesianGrids.Goza)

  # NEED A MULTIBODY VERSION OF THIS (OR ASSUME MULTIBODY)

  #coords = VectorData(collect(bl))
  coords = VectorData(body.x,body.y)

  sys = setup_system(Re,Δx,xlim,ylim,Δt,coords;ddftype=ddftype)

  u = initialize_state(sys)
  f = initialize_force(sys)

  #=
  The integrating factor for the first two equations is simply the one associated
  with the usual viscous term. The last two equations have no term that needs an
  integrating factor, so we set their integrating factor operators to the identity, I.
  =#
  plans = ((t,u) -> CartesianGrids.plan_intfact(t,u,sys),(t,u) -> CartesianGrids.plan_intfact(t,u,sys),
           (t,u) -> Identity(),(t,u)-> I)

  gradopx = Regularize(sys.X̃,cellsize(sys);I0=origin(sys),issymmetric=true,ddftype=ddftype,graddir=1)
  gradopy = Regularize(sys.X̃,cellsize(sys);I0=origin(sys),issymmetric=true,ddftype=ddftype,graddir=2)
  dEx = InterpolationMatrix(gradopx,sys.Fq,sys.Vb)
  dEy = InterpolationMatrix(gradopy,sys.Fq,sys.Vb)

  # adapt these to multibody!
  streaming_r₁(u,t) = r₁(u,t,sys,motion)
  streaming_r₂(u,t) = r₂(u,t,sys,motion,body.cent,dEx,dEy)
  streaming_plan_constraints(u,t::Float64) = plan_constraints(u,t,sys)

  return IFHERK(u,f,sys.Δt,plans,streaming_plan_constraints,
                      (streaming_r₁,streaming_r₂),rk=ConstrainedSystems.RK31,isstored=true), sys

end

#=
multibody version
=#
function initialize_solver(Re,Δx,xlim,ylim,Δt,bl::BodyList,ml::Vector{RigidBodyMotion},tl::Vector{RigidTransform};ddftype=CartesianGrids.Goza)

  coords = VectorData(collect(bl))

  sys = setup_system(Re,Δx,xlim,ylim,Δt,coords;ddftype=ddftype)

  u = initialize_state(sys,length(bl))
  f = initialize_force(sys)

  #=
  The integrating factor for the first two equations is simply the one associated
  with the usual viscous term. The last two equations have no term that needs an
  integrating factor, so we set their integrating factor operators to the identity, I.
  =#
  plans = ((t,u) -> CartesianGrids.plan_intfact(t,u,sys),(t,u) -> CartesianGrids.plan_intfact(t,u,sys),
           (t,u) -> Identity(),(t,u)-> I)

  gradopx = Regularize(sys.X̃,cellsize(sys);I0=origin(sys),issymmetric=true,ddftype=ddftype,graddir=1)
  gradopy = Regularize(sys.X̃,cellsize(sys);I0=origin(sys),issymmetric=true,ddftype=ddftype,graddir=2)
  dEx = InterpolationMatrix(gradopx,sys.Fq,sys.Vb)
  dEy = InterpolationMatrix(gradopy,sys.Fq,sys.Vb)

  streaming_r₁(u,t) = r₁(u,t,sys,ml)
  streaming_r₂(u,t) = r₂(u,t,sys,bl,ml,tl,dEx,dEy)
  streaming_plan_constraints(u,t::Float64) = plan_constraints(u,t,sys)

  return IFHERK(u,f,sys.Δt,plans,streaming_plan_constraints,
                      (streaming_r₁,streaming_r₂),rk=ConstrainedSystems.RK31,isstored=true), sys

end

#=
The right-hand side of the first-order equation is 0. The right-hand side of the
second-order equation is the negative of the non-linear convective term, based on
the first-order solution; for this, we use the predefined `r₁`. The right-hand side
of the Eulerian displacement field equation is just the corresponding velocity at that
level. The right-hand side of the body update equation is the unscaled velocity.
=#
function r₁(u::TU,t::Float64,sys::NavierStokes,motion::RigidBodyMotion)
    _,ċ,_,_,α̇,_ = motion(t)
    return zero(u[1]), r₁(u[1],t,sys), lmul!(-1,curl(sys.L\u[1])), [real(ċ),imag(ċ)]
end

#=
For multiple bodies
=#
function r₁(u::TU,t::Float64,sys::NavierStokes,ml::Vector{RigidBodyMotion})
    return zero(u[1]), r₁(u[1],t,sys), -curl(sys.L\u[1]), r₁(u[4],t,ml)
end

#=
The right-hand side of the first constraint equation is the unscaled rigid-body velocity,
evaluated at the surface points. The right-hand side of the second constraint equation
is $-\hat{X}\cdot\nabla v_1$. The Eulerian displacement and the body update equations
have no constraint, so these are set to an empty vector.
=#
function r₂(u::TU,t::Float64,sys::NavierStokes,motion::RigidBodyMotion,centroid,dEx,dEy)
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

#=
This is for multibody
=#
function r₂(u::TU,t::Float64,sys::NavierStokes,bl::BodyList,ml::Vector{RigidBodyMotion},tl::Vector{RigidTransform},dEx,dEy)
  fact = 2 # not sure how to explain this factor yet.

  Xc = zero(sys.Vb) # instantaneous centroid
  Xc0 = zero(sys.Vb) # mean centroid
  U = zero(sys.Vb)  # translational velocity
  α̇ = zero(sys.Vb.u) # angular velocity
  for ib = 1:length(ml)
      _,ċ,_,_,α̇i,_ = ml[ib](t)

      ri = getrange(bl,ib)

      # fill VectorData with translational velocity
      U.u[ri] .= real(ċ)
      U.v[ri] .= imag(ċ)
      α̇[ri] .= α̇i

      Xci = vec(tl[ib])
      Xc0.u[ri] .= Xci[1]
      Xc0.v[ri] .= Xci[2]

      # fill VectorData with corresponding instantaneous centroids
      Xc.u[ri] .= u[4][3*(ib-1)+1]
      Xc.v[ri] .= u[4][3*(ib-1)+2]
    end

  # -ΔX̂⋅∇v₁
  Δx⁻¹ = 1/cellsize(sys)

  sys.Fq .= curl(sys.L\u[1]) # -v₁
  sys.Vb .= dEx*sys.Fq # dv₁/dx*Δx
  sys.Vb.u .*= -fact*Δx⁻¹*Xc.u
  sys.Vb.v .*= -fact*Δx⁻¹*Xc.u
  Vb = deepcopy(sys.Vb) # -X⋅dv₁/dx
  sys.Vb .= dEy*sys.Fq # dv₁/dy*Δx
  sys.Vb.u .*= -fact*Δx⁻¹*Xc.v
  sys.Vb.v .*= -fact*Δx⁻¹*Xc.v
  Vb .+= sys.Vb # -X⋅dv₁/dx - Y.⋅dv₁/dy
  return U + α̇ × (sys.X̃ - Xc0), Vb, Vector{Float64}(), Vector{Float64}()

end

#=
The constraint operators for the first two equations are the usual ones for a
stationary body and are precomputed. There are no constraints or constraint forces
for the last three equations.
=#
function plan_constraints(u::TU,t::Float64,sys::NavierStokes)
    # These are used by both the first and second equations
    B₁ᵀ, B₂ = plan_constraints(u[1],t,sys)
    return (B₁ᵀ,B₁ᵀ,f->zero(u[3]),f->zero(u[4])),
            (B₂,B₂,u->Vector{Float64}(),u->Vector{Float64}())
end
