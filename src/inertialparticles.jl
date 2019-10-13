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
          {S<:XEdges,T<:YEdges,R<:Nodes,H<:Systems.HistoryType}

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
    a = accelforce(u,dudt,g,p.β,p.Re)
    return u + p.τ*a - sqrt(p.ϵ*p.β*p.τ^3)*saffman(a,ω)
end

"""
    inertial_velocity(u::Edges{Primal},dudt::Edges{Primal},g::PhysicalGrid,p::InertialParameters)

Return the inertial particle velocity field
```math
\\mathbf{v} = \\mathbf{u} + \\tau \\mathbf{a}
```
for a given fluid velocity field `u` and its time derivative `dudt`. The physical grid data is given in `g`, and
the physical parameters are also supplied in `p`. The result is returned as
primal edge data of the same size as `u`.
"""
function inertial_velocity(u::Edges,dudt::Edges,g::PhysicalGrid,p::InertialParameters)
    a = accelforce(u,dudt,g,p.β,p.Re)
    return u + p.τ*a
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
    ViscousFlow.interpolate!(Ls.u,ViscousFlow.interpolate!(uynode, u.v) ∘ ω)
    ViscousFlow.interpolate!(Ls.v,ViscousFlow.interpolate!(uxnode,-u.u) ∘ ω)

    ωx = zero(u.u)
    ωy = zero(u.v)
    ViscousFlow.interpolate!(ωx,ω)  # vorticity on primal x edges
    ViscousFlow.interpolate!(ωy,ω)  # vorticity on primal y edges

    #Ls.u .*= 3sqrt(3)/(2π^2)*J∞./(ωx.^2 .+ 1e-15).^(1/4)
    #Ls.v .*= 3sqrt(3)/(2π^2)*J∞./(ωy.^2 .+ 1e-15).^(1/4)
    Ls.u .*= 3sqrt(3)/(2π^2)*J∞./sqrt.(abs.(ωx).+1e-8)
    Ls.v .*= 3sqrt(3)/(2π^2)*J∞./sqrt.(abs.(ωy).+1e-8)

    return Ls
end

#=
Time derivatives
=#

ddt(u::History{T,PeriodicHistory}) = 0.5*(diff(u) + diff(circshift(u,1)))

ddt(u,Δt::Real) = ddt(u)/Δt
