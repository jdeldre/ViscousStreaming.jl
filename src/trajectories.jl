# Routines for computing particle trajectories

using OrdinaryDiffEq
using DiffEqCallbacks # for manifold projection
using Interpolations

export compute_trajectory

function compute_trajectory(vfcn::Function,X₀::Tuple,Tmax::Real,Δt::Real;bl::BodyList=BodyList(),ϵ::Real=0.0)

  u0 = [X₀[1],X₀[2]]
  tspan=(0.0,Tmax)

  if length(bl) > 0
    indicator!(resid,u,p,t) = body_indicator!(resid,u,p,t,ϵ,bl)
    cbdomain = ManifoldProjection(indicator!)
    Path = ODEProblem(vfcn,u0,tspan,callback=cbdomain)
    sol = solve(Path,ABM54(), dt = Δt, maxiters = 1e8, adaptive = false, dense = true, progressbar = true);
  else
    Path = ODEProblem(vfcn,u0,tspan)
    sol = solve(Path,ABM54(), dt = Δt, maxiters = 1e8, adaptive = false, dense = false, progressbar = true);
  end

  return sol
end

compute_trajectory(vfcn::Function,X₀::Tuple,Tmax::Real,Δt::Real,body::Body;ϵ::Real=0.0) =
      compute_trajectory(vfcn,X₀,Tmax,Δt;bl=BodyList([body]),ϵ=ϵ)




# This is the function that is constrained to be zero by manifold projection
function body_indicator!(resid,u,p,t,ϵ,bl)
  # this only works for circles now
  maxϕ = -1.0e10
  for b in bl
    maxϕ = max(maxϕ,ϕ(u,[b.cent[1],b.cent[2]],b.a))
  end
  resid[1] = Hs(ϵ)(maxϕ)
  resid[2] = 0.0
end

#
function ϕ(x,xc::Vector,R::Real)
    # Returns a value that is positive if x is inside a circle of radius R
    # or negative if x is outside the circle
    out = R - norm(x-xc)
    return out
end


# Heaviside functions
H(x::AbstractFloat) = ifelse(x < 0, zero(x), one(x))

Hs!(x,δ) = ifelse(x < -δ, zero(x), ifelse(x > δ, one(x), 0.5*(1+x/δ+1/π*sin(π*x/δ))))

Hs(β)=x->Hs!(x,β)
