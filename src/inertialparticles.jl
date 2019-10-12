# Routines associated with computing the inertial particle velocity field
# from the fluid velocity field

export acceleration_force

"""
    acceleration_force(u::Edges,dudt::Edges,g::PhysicalGrid,β::Real,Re::Real)

Calculate the acceleration force
```math
\\mathbf{a} = (\\beta-1)\\dfrac{d\\mathbf{u}}{dt} + \\frac{\\beta}{2Re} \\nabla^2 \\mathbf{u}
```
from the given velocity data `u` and associated time derivative `dudt`. Note
that `dudt` might represent simply the partial derivative or the material derivative,
depending on what is passed. The grid data in `g` is used for the grid spacing.
"""
function acceleration_force(q::T,dqdt::T,g::PhysicalGrid,β::Real,Re::Real) where {T <: Edges}
    return (β-1)*dqdt + 0.5β/Re*laplacian(q)/cellsize(g)^2
end
