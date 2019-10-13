# Routines associated with computing the inertial particle velocity field
# from the fluid velocity field

export InertialParameters,inertial_velocity, acceleration_force, saffman

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
    inertial_velocity(u::Edges{Primal},dudt::Edges{Primal},ω::Nodes{Dual},g,p::InertialParameters)

Return the inertial particle velocity field
```math
\\mathbf{v} = \\mathbf{u} + \\tau \\mathbf{a} - \\epsilon^{1/2} \\beta^{1/2} \\tau^{3/2} \\mathcal{L}_s(\\mathbf{a},\\mathbf{\\omega})
```
for a given fluid velocity field `u`, its
time derivative `dudt`, and vorticity field `ω`. The physical grid data is given in `g`, and
the physical parameters `β`, `Re`, `τ` and `ϵ` are also supplied. The result is returned as
primal edge data of the same size as `u`.
"""
function inertial_velocity(u::Edges,dudt::Edges,ω::Nodes,g::PhysicalGrid,p::InertialParameters)
    a = accelforce(u,dudt,g,p.β,p.Re)
    return u + p.τ*a - sqrt(p.ϵ*p.β*p.τ^3)*saffman(a,ω)
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

    ωx = deepcopy(u.u)
    ωy = deepcopy(u.v)
    ViscousFlow.interpolate!(ωx,ω)  # vorticity on primal x edges
    ViscousFlow.interpolate!(ωy,ω)  # vorticity on primal y edges

    #Ls.u .*= 3sqrt(3)/(2π^2)*J∞./(ωx.^2 .+ 1e-15).^(1/4)
    #Ls.v .*= 3sqrt(3)/(2π^2)*J∞./(ωy.^2 .+ 1e-15).^(1/4)
    Ls.u .*= 3sqrt(3)/(2π^2)*J∞./sqrt.(abs.(ωx).+1e-8)
    Ls.v .*= 3sqrt(3)/(2π^2)*J∞./sqrt.(abs.(ωy).+1e-8)

    return Ls
end
