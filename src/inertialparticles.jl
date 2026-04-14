# Routines associated with computing the inertial particle velocity field
# from the fluid velocity field

using Interpolations

import ImmersedLayers: laplacian!

export InertialParameters,inertial_velocity, _unscaled_convective_derivative!, ParticleFlow

# for testing only
export acc1, acc2, saffman

"""
    InertialParameters

Set the inertial particle transport parameters ``\\beta``, ``\\tau``, ``\\epsilon``,
and ``Re``. The constructor uses keyword arguments with these written out.

# Example

```jldoctest
julia> p = InertialParameters(tau=0.1,beta=0.95,epsilon=0.1,Re=40)
```
"""
struct InertialParameters
  β :: Real
  τ :: Real
  ϵ :: Real
  Re :: Real
end

InertialParameters(;beta,tau,epsilon,Re) = InertialParameters(beta,tau,epsilon,Re)

# The version in CartesianGrids is too restrictive, so include this less
# restrictive one here
@inline product!(out::Nodes{C,NX,NY,F}, p::Nodes{C,NX,NY,F},
                  q::Nodes{C,NX,NY,F}) where {C,NX,NY,F} = (out .= p.*q)

@inline product!(out::EdgeGradient{C,NX,NY,F},
                 p::EdgeGradient{C,NX,NY,F},
                 q::EdgeGradient{C,NX,NY,F}) where {C,NX,NY,F} = (out .= p .* q)
            
function product!(out::VectorData, a::VectorData, b::ScalarData)
    @. out.u = a.u * b
    @. out.v = a.v * b
    return out
end

#=
"""
    inertial_velocity(ux::History{XEdges},uy::History{YEdges},
        duxdt::History{XEdges},duydt::History{YEdges},w::History{Nodes},
        g,Δt,p::InertialParameters)

Calculate the time history of the inertial particle velocity field for a given fluid velocity
field, whose x and y component histories are given in `ux` and `uy` and whose
vorticity (unscaled by grid spacing) is given in `w`. The physical grid data is given in `g`,
the time step size corresponding to the histories is in `Δt`, and the physical parameters in
`p`. The result is returned as a tuple of `History{XEdge}` and `History{YEdge}`.
"""

function inertial_velocity(ux::History{S,H},uy::History{T,H},
        duxdt::History{S,H},duydt::History{T,H},
        w::History{R,H},
        g::PhysicalGrid,Δt::Real,p::InertialParameters;cflag::Bool=false) where
          {S<:XEdges,T<:YEdges,R<:Nodes,H<:HistoryType}

    # vx, vy serve as temp storage for du/dt
    vx = deepcopy(duxdt)
    vy = deepcopy(duydt)

    u = Edges(Primal,w[1])
    u2 = Edges(Primal,w[1])

    for (i,wi) in enumerate(w)

        # add u.grad(u) to du/dt
        if cflag
            u .= typeof(u)(ux[i],uy[i])
            directional_derivative!(u2,u,u)

            u2 ./= cellsize(g)
            vx[i] .+= u2.u
            vy[i] .+= u2.v
        end

        u .= inertial_velocity(Edges(ux[i],uy[i]),Edges(vx[i],vy[i]),vorticity(wi,g),g,p)
        vx[i] .= u.u
        vy[i] .= u.v
    end
    return vx, vy
end

"""
    inertial_velocity(ux::History{XEdges},uy::History{YEdges},
                            u1x::History{XEdges},u1y::History{YEdges},
                            duxdt::History{XEdges},duydt::History{YEdges},
                            du1xdt::History{XEdges},du1ydt::History{YEdges},w::History{Nodes},g,Δt,p)

Calculate the first two terms of the asymptotically-expanded time history of the inertial
particle velocity field for a given fluid velocity
field, whose x and y component histories are given in `ux` and `uy`, whose first-order
velocities are `u1x` and `u1y`, and whose Eulerian time derivatives of these are
`duxdt`, `duydt`, `du1xdt`, `du1ydt`, and whose
vorticity (unscaled by grid spacing) is given in `w`. The physical grid data is given in `g`,
the time step size corresponding to the histories is in `Δt`, and the physical parameters in
`p`. The result is returned as a tuple of `History{XEdge}` and `History{YEdge}`.
"""
function inertial_velocity(ux::History{S,H},uy::History{T,H},
        u1x::History{S,H},u1y::History{T,H},
        duxdt::History{S,H},duydt::History{T,H},
        du1xdt::History{S,H},du1ydt::History{T,H},
        w::History{R,H},
        g::PhysicalGrid,Δt::Real,p::InertialParameters;cflag::Bool=false) where {S<:XEdges,T<:YEdges,R<:Nodes,H<:HistoryType}

    # vx, vy serve as temp storage for du/dt
    vx = deepcopy(duxdt)
    vy = deepcopy(duydt)

    u = Edges(Primal,w[1])
    u2 = Edges(Primal,w[1])

    for (i,wi) in enumerate(w)

        # add u1.grad(u1) to du/dt
        if cflag
            u .= typeof(u)(u1x[i],u1y[i])
            directional_derivative!(u2,u,u)

            u2 ./= cellsize(g)
            vx[i] .+= u2.u
            vy[i] .+= u2.v
        end
        a1 = acceleration_force(Edges(u1x[i],u1y[i]),Edges(du1xdt[i],du1ydt[i]),g,p)

        u .= inertial_velocity(Edges(ux[i],uy[i]),Edges(vx[i],vy[i]),g,p)
        u .-= sqrt(p.ϵ*p.β*p.τ^3)*saffman(a1,vorticity(wi,g))
        vx[i] .= u.u
        vy[i] .= u.v
    end
    return vx, vy
end
=#

#=
"""
    inertial_velocity(u::Edges{Primal},dudt::Edges{Primal},ω::Nodes{Dual},g::PhysicalGrid,p::InertialParameters)

Return the inertial particle velocity field
```math
\\mathbf{v} = \\mathbf{u} + \\tau \\mathbf{a} - \\epsilon^{1/2} \\beta^{1/2} \\tau^{3/2} \\mathcal{L}_s(\\mathbf{a},\\mathbf{\\omega})
```
for a given fluid velocity field `u`, its
time derivative `dudt`, and vorticity field `ω`. Note that `ω` should be the proper
vorticity (i.e., scaled by the grid spacing). The physical grid data is given in `g`, and
the physical parameters are also supplied in `p`. The result is returned as
primal edge data of the same size as `u`.
"""
function inertial_velocity(u::Edges,dudt::Edges,ω::Nodes,g::PhysicalGrid,p::InertialParameters)
    a = acceleration_force(u,dudt,g,p)
    return u + p.τ*a - sqrt(p.ϵ*p.β*p.τ^3)*saffman(a,ω)
end

"""
    inertial_velocity(u::Edges{Primal},dudt::Edges{Primal},g::PhysicalGrid,p::InertialParameters)

Return the inertial particle velocity field without Saffman lift term,
```math
\\mathbf{v} = \\mathbf{u} + \\tau \\mathbf{a}
```
for a given fluid velocity field `u` and its time derivative `dudt`. The physical grid data is given in `g`, and
the physical parameters are also supplied in `p`. The result is returned as
primal edge data of the same size as `u`.
"""
function inertial_velocity(u::Edges,dudt::Edges,g::PhysicalGrid,p::InertialParameters)
    a = acceleration_force(u,dudt,g,p)
    return u + p.τ*a
end
=#

#=
"""
"""
function inertial_velocity(s::StreamingComputational{FluidFlow},p::InertialParameters)

  NX, NY = size(s.s1)

  # Compute first-order velocity
  u1 = s.s1.U
  du1dt = ddt(s.s1).U
  a1 = acceleration_force(u1,du1dt,s.g,p)

  v1 = u1 + p.τ*a1

  soln1 = AsymptoticComputational{FirstOrder,ParticleFlow,NX,NY}(s.p.Re,s.p.ϵ,s.p.Ω,s.g,
                                            nothing,nothing,v1)

  # second-order mean velocity, with mean Saffman term
  Ls0, Ls2 = saffman(a1,s.s1.W)

  # Add u1.grad u1 to dudt
  u1gradu1 = deepcopy(s.s1.U)
  directional_derivative!(u1gradu1,conj(u1),u1)
  dū2dt = 0.5/cellsize(s.g)*u1gradu1

  ū2 = s.s̄2.U
  ā2 = acceleration_force(ū2,dū2dt,s.g,p)
  v̄2 = ū2 + p.τ*ā2 - sqrt(p.β*p.τ^3/p.ϵ)*Ls0

  meansoln2 = AsymptoticComputational{SecondOrderMean,ParticleFlow,NX,NY}(s.p.Re,s.p.ϵ,s.p.Ω,s.g,
                                            nothing,nothing,v̄2)

  # mean drift velocity v1.grad v1*
  directional_derivative!(u1gradu1,conj(v1),v1)
  v̄d = (-0.5im/s.p.Ω/cellsize(s.g))*u1gradu1
  sdsoln = AsymptoticComputational{SecondOrderMean,ParticleFlow,NX,NY}(s.p.Re,s.p.ϵ,s.p.Ω,s.g,
                                            nothing,nothing,v̄d)


  return StreamingComputational{ParticleFlow}(s.p,s.g,soln1,meansoln2,sdsoln,nothing)
end
=#

#=
"""
    acceleration_force(u::Edges,dudt::Edges,g::PhysicalGrid,p::InertialParameters)

Calculate the acceleration force
```math
\\mathbf{a} = (\\beta-1)\\dfrac{d\\mathbf{u}}{dt} + \\frac{\\beta}{2Re} \\nabla^2 \\mathbf{u}
```
from the given velocity data `u` and associated time derivative `dudt`. Note
that `dudt` might represent simply the partial derivative or the material derivative,
depending on what is passed. The grid data in `g` is used for the grid spacing.
"""
function acceleration_force(u::T,dudt::T,g::PhysicalGrid,p::InertialParameters) where {T <: Edges}
    return (p.β-1)*dudt + 0.5p.β/p.Re*laplacian(u)/cellsize(g)^2
end
=#

"""
    acc1(u::Edges,dudt::Edges,g::PhysicalGrid,p::InertialParameters)

Calculate the first-order acceleration force from the given velocity data `u1` as complex amplitude
"""
function acc1(u1::Edges{Primal,NX,NY,ComplexF64},cache1::BasicILMCache,p::InertialParameters) where {NX,NY}
    lu1 = zeros_grid(cache1)
    laplacian!(lu1, u1, cache1)
    a1 = im * (p.β-1) * u1 + 0.5 * p.β / p.Re * lu1;
    return a1
end

"""
    acc2(u::Edges,dudt::Edges,g::PhysicalGrid,p::InertialParameters)

Calculate the second-order acceleration force from the given mean velocity data `u2`
"""
function acc2(u1::Edges{Primal,NX,NY,ComplexF64}, u2::Edges{Primal,NX,NY,Float64},cache1::BasicILMCache, cache2::BasicILMCache, p::InertialParameters) where {NX,NY}
    udu = zeros_grid(cache1);
    cdcache = ConvectiveDerivativeCache(cache1);
    _unscaled_convective_derivative!(udu, u1, conj(u1), cdcache)
    ImmersedLayers._scale_derivative!(udu, cache1)
    lu2 = zeros_grid(cache2)
    laplacian!(lu2, u2, cache2)
    a2 = 0.5 * (p.β-1) * real(udu) + 0.5 * p.β / p.Re * lu2;
    return a2
end

#=
"""
    saffman(u::Edges{Primal},ω::Nodes{Dual})

Computes the Saffman lift operator ``\\mathcal{L}_s``, using velocity field `u` (in primal edge data)
and vorticity field `ω` (in dual node data). Note that `ω` should be the proper
vorticity (i.e., scaled by the grid spacing). The result is returned as primal edge data.
"""
function saffman(u::Edges{Primal},ω::Nodes{Dual})
    J∞ = 2.255
    Ls = zero(u)

    uxnode = Nodes(Dual,ω)
    uynode = Nodes(Dual,ω)
    grid_interpolate!(Ls.u,grid_interpolate!(uynode, u.v) ∘ ω)
    grid_interpolate!(Ls.v,grid_interpolate!(uxnode,-u.u) ∘ ω)

    ωx = zero(u.u)
    ωy = zero(u.v)
    grid_interpolate!(ωx,ω)  # vorticity on primal x edges
    grid_interpolate!(ωy,ω)  # vorticity on primal y edges

    #Ls.u .*= 3sqrt(3)/(2π^2)*J∞./(ωx.^2 .+ 1e-15).^(1/4)
    #Ls.v .*= 3sqrt(3)/(2π^2)*J∞./(ωy.^2 .+ 1e-15).^(1/4)
    Ls.u .*= 3sqrt(3)/(2π^2)*J∞./sqrt.(abs.(ωx).+1e-8)
    Ls.v .*= 3sqrt(3)/(2π^2)*J∞./sqrt.(abs.(ωy).+1e-8)

    return Ls
end
=#

"""
    ParticleFlowField(v1, v2, vL, vL_u_interp, vL_v_interp)

A struct to hold the inertial particle velocity field information, including the first and second order inertial velocities (v1, v2), the Lagrangian mean velocity (vL), and the interpolatable fields for the Lagrangian mean velocity components (vL_u_interp, vL_v_interp).
"""
struct ParticleFlowField
    v1
    v2
    v_L
    v_L_u_interp
    v_L_v_interp
end

"""
    saffman(u, ω; return_debug=false, ω_floor=1e-10)

Computes the Saffman lift operator. If `return_debug=true`, also returns a NamedTuple
of intermediate quantities for debugging/comparison.
"""
function saffman(
    u::Edges{Primal,NX,NY,ComplexF64},
    ω::Nodes{Dual,NX,NY,ComplexF64};
    return_debug::Bool=false,
    ω_floor::Float64=1e-10
) where {NX,NY}

    J∞ = 2.255
    Ω = 1.0

    Ĉ0 = _coefficient(0)
    Ĉ2 = _coefficient(2)

    absω = similar(ω, Float64)
    absω .= abs.(ω)

    safe_absω = similar(ω, Float64)
    safe_absω .= max.(absω, ω_floor)

    expiϕ = similar(ω)
    expiϕ .= ω ./ safe_absω

    b0_node = similar(ω)
    b2_node = similar(ω)

    b0_node .= Ĉ0
    b2_node .= Ĉ2 * (expiϕ ∘ expiϕ)

    b0_node ./= sqrt.(safe_absω)
    b2_node ./= sqrt.(safe_absω)

    b0 = similar(u)
    b2 = similar(u)

    grid_interpolate!(b0.u, b0_node)
    grid_interpolate!(b0.v, b0_node)
    grid_interpolate!(b2.u, b2_node)
    grid_interpolate!(b2.v, b2_node)

    a0 = similar(u)
    a2 = similar(u)

    uxnode = Nodes(Dual, ω, dtype=ComplexF64)
    uynode = Nodes(Dual, ω, dtype=ComplexF64)
    grid_interpolate!(uynode, u.v)
    grid_interpolate!(uxnode, -u.u)

    a0_node_u = similar(ω)
    a0_node_v = similar(ω)
    a2_node_u = similar(ω)
    a2_node_v = similar(ω)

    a0_node_u .= uynode ∘ conj(ω)
    a0_node_v .= uxnode ∘ conj(ω)
    a2_node_u .= uynode ∘ ω
    a2_node_v .= uxnode ∘ ω

    grid_interpolate!(a0.u, a0_node_u)
    grid_interpolate!(a0.v, a0_node_v)
    grid_interpolate!(a2.u, a2_node_u)
    grid_interpolate!(a2.v, a2_node_v)

    Ls0 = 0.5 * (a0 ∘ b0 + conj(a0) ∘ b0 + conj(a2) ∘ b2)
    Ls2 = 0.5 * (a0 ∘ b2 + conj(a0) ∘ b2 + b0 ∘ a2 + conj(b0) ∘ a2)

    K = 3sqrt(3) / (2π^2) * J∞ / Ω
    Ls0 .*= K
    Ls2 .*= K

    if return_debug
        debug = (
            Ĉ0 = Ĉ0,
            Ĉ2 = Ĉ2,
            ω = deepcopy(ω),
            absω = deepcopy(absω),
            safe_absω = deepcopy(safe_absω),
            expiϕ = deepcopy(expiϕ),
            b0_node = deepcopy(b0_node),
            b2_node = deepcopy(b2_node),
            b0 = deepcopy(b0),
            b2 = deepcopy(b2),
            uxnode = deepcopy(uxnode),
            uynode = deepcopy(uynode),
            a0_node_u = deepcopy(a0_node_u),
            a0_node_v = deepcopy(a0_node_v),
            a2_node_u = deepcopy(a2_node_u),
            a2_node_v = deepcopy(a2_node_v),
            a0 = deepcopy(a0),
            a2 = deepcopy(a2),
            Ls0 = deepcopy(Ls0),
            Ls2 = deepcopy(Ls2),
            K = K,
        )
        return Ls0, Ls2, debug
    else
        return Ls0, Ls2
    end
end

_coefficient(n) = 2^(1/2)*(ellipk(1/2)/π)*gamma(n/2+1/4)^2/gamma(1/4)^2/gamma(n+1/2)*sqrt(π)*2^n*(-1)^(n/2)

#=
Time derivatives
=#

#=
ddt(u::History{T,PeriodicHistory}) where {T} = 0.5*(diff(u) + diff(circshift(u,1)))

ddt(u,Δt::Real) = ddt(u)/Δt

function ddt(s::AsymptoticComputational{FirstOrder,F,NX,NY}) where {F,NX,NY}
    return AsymptoticComputational{FirstOrder,F,NX,NY}(s.Re,s.ϵ,s.Ω,s.g,
                      im*s.Ω*s.W,im*s.Ω*s.Ψ,im*s.Ω*s.U)
end

function ddt(s::AsymptoticComputational{SecondOrder,F,NX,NY}) where {F,NX,NY}
    return AsymptoticComputational{SecondOrder,F,NX,NY}(s.Re,s.ϵ,s.Ω,s.g,
                      2im*s.Ω*s.W,2im*s.Ω*s.Ψ,2im*s.Ω*s.U)
end
=#

#=
Frequency domain routines
=#

"""
    inertial_velocity(u1, u2, ω1, cache1, cache2, p)
    inertial_velocity(flowfield, p)

Compute the first- and second-order contributions to the inertial particle
velocity field in the frequency domain.

# Methods
- `inertial_velocity(u1, u2, ω1, cache1, cache2, p)`:
  Core implementation. Uses provided velocity fields, vorticity, and ILM caches.

- `inertial_velocity(flowfield, p)`:
  Convenience wrapper. Extracts `u1`, `u2`, `ω1` and constructs caches from
  `flowfield`, then calls the core method.

# Arguments
- `u1`: First-order velocity field (complex)
- `u2`: Second-order mean velocity field (real)
- `ω1`: First-order vorticity field (complex)
- `cache1`, `cache2`: ILM caches for acceleration computation
- `flowfield`: Struct containing `u1`, `u2`, `ω1`, grid, and body
- `p`: Inertial parameters

# Returns
- `(v1, v2)`: First- and second-order inertial particle velocities
"""
function inertial_velocity(u1::Edges{Primal,NX,NY,ComplexF64}, u2::Edges{Primal,NX,NY,Float64}, ω1::Nodes{Dual,NX,NY,ComplexF64}, cache1::BasicILMCache, cache2::BasicILMCache, p::InertialParameters) where {NX,NY}
    a1 = acc1(u1,cache1,p)
    v1 = u1 + p.τ*a1
    a2 = acc2(u1, u2, cache1, cache2, p)
    Ls0, Ls2 = saffman(a1,ω1)
    v2 = u2 + p.τ*a2 - sqrt(p.β*p.τ^3/p.ϵ)*real(Ls0)

    vdv = zeros_grid(cache1)
    cdcache = ConvectiveDerivativeCache(cache1);
    _unscaled_convective_derivative!(vdv, v1, conj(v1), cdcache)
    ImmersedLayers._scale_derivative!(vdv, cache1)
    v_drift =  1/2 * imag(vdv); 

    v_L = v2 + v_drift;
    v_L_u_interp = interpolatable_field(v_L.u, cache1.g)
    v_L_v_interp = interpolatable_field(v_L.v, cache1.g)
    return ParticleFlowField(v1, v2, v_L, v_L_u_interp, v_L_v_interp)
end

function inertial_velocity(flowfield::FlowField, p::InertialParameters)
    # set up the cache 
    g = flowfield.g
    body = flowfield.body
    cache1 = SurfaceVectorCache(body, g, dtype=ComplexF64);
    cache2 = SurfaceVectorCache(body, g, dtype=Float64);

    # extract flow states
    u1 = zeros_grid(cache1);
    u2 = zeros_grid(cache2);
    ω1 = zeros_gridcurl(cache1)
    u1 .= flowfield.u1
    u2 .= flowfield.u2
    ω1 .= flowfield.ω1

    return inertial_velocity(u1, u2, ω1, cache1, cache2, p)
end
