# Routines associated with computing the inertial particle velocity field
# from the fluid velocity field

export InertialParameters,inertial_velocity, acceleration_force, saffman, ddt

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



"""
    saffman(u::Edges{Primal,NX,NY,ComplexF64},ω::Nodes{DualNX,NY,ComplexF64})

Computes the Saffman lift operator ``\\mathcal{L}_s``, using velocity field `u` (in primal edge data)
and vorticity field `ω` (in dual node data), given as complex amplitudes for oscillatory
solution. Note that `ω` should be the proper vorticity (i.e., scaled by the grid spacing).
The result (the 0th and 2nd-order Fourier coefficients) are returned as primal edge data
"""
function saffman(u::Edges{Primal,NX,NY,ComplexF64},ω::Nodes{Dual,NX,NY,ComplexF64}) where {NX,NY}

  J∞ = 2.255
  Ω = 1.0

  Ĉ0 = _coefficient(0)
  Ĉ2 = _coefficient(2)

  expiϕ = similar(ω)
  expiϕ .= ω./abs(ω)

  b0_node = similar(ω)
  b2_node = similar(ω)

  b0_node .= Ĉ0
  b2_node .= Ĉ2*(expiϕ∘expiϕ)

  b0_node ./= sqrt.(abs(ω))
  b2_node ./= sqrt.(abs(ω))

  b0 = similar(u)
  b2 = similar(u)

  grid_interpolate!(b0.u,b0_node)  # primal x edges
  grid_interpolate!(b0.v,b0_node)
  grid_interpolate!(b2.u,b2_node)  # primal y edges
  grid_interpolate!(b2.v,b2_node)

  a0 = similar(u)
  a2 = similar(u)

  uxnode = Nodes(Dual,ω,dtype=ComplexF64)
  uynode = Nodes(Dual,ω,dtype=ComplexF64)
  grid_interpolate!(uynode, u.v)
  grid_interpolate!(uxnode,-u.u)

  grid_interpolate!(a0.u, uynode ∘ conj(ω))
  grid_interpolate!(a0.v, uxnode ∘ conj(ω))
  grid_interpolate!(a2.u, uynode ∘ ω)
  grid_interpolate!(a2.v, uxnode ∘ ω)

  Ls0 = 0.5*(a0 ∘ b0 + conj(a0) ∘ b0 + conj(a2) ∘ b2)
  Ls2 = 0.5*(a0 ∘ b2 + conj(a0) ∘ b2 + b0 ∘ a2 + conj(b0) ∘ a2)

  K = 3sqrt(3)/(2π^2)*J∞/Ω

  Ls0 .*= K
  Ls2 .*= K

  return Ls0, Ls2

end

_coefficient(n) = 2^(1/2)*(ellipk(1/2)/π)*gamma(n/2+1/4)^2/gamma(1/4)^2/gamma(n+1/2)*sqrt(π)*2^n*(-1)^(n/2)

#=
Time derivatives
=#

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

#=
Frequency domain routines
=#
