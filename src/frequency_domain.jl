#=
Routines for the frequency domain solution
=#

import Base: size

abstract type StreamingSystem{NX,NY,N} end

# some convenience functions, which should be defined generally in ViscousFlow
# for all systems.
"""
    size(sys::StreamingSystem,d::Int) -> Int

Return the number of indices of the grid used by `sys` along dimension `d`.
"""
size(sys::StreamingSystem{NX,NY},d::Int) where {NX,NY} = d == 1 ? NX : NY

"""
    size(sys::StreamingSystem) -> Tuple{Int,Int}

Return a tuple of the number of indices of the grid used by `sys`
"""
size(sys::StreamingSystem{NX,NY}) where {NX,NY} = (size(sys,1),size(sys,2))

"""
    cellsize(sys::StreamingSystem) -> Float64

Return the grid cell size of system `sys`
"""
Fields.cellsize(sys::StreamingSystem) = cellsize(sys.grid)

"""
    origin(sys::StreamingSystem) -> Tuple{Int,Int}

Return a tuple of the indices of the primal node that corresponds to the
physical origin of the coordinate system used by `sys`. Note that these
indices need not lie inside the range of indices occupied by the grid.
For example, if the range of physical coordinates occupied by the grid
is (1.0,3.0) x (2.0,4.0), then the origin is not inside the grid.
"""
Fields.origin(sys::StreamingSystem) = origin(sys.grid)


"""
    FrequencyStreaming

Sets up the solution operator for frequency-based viscous streaming solution.
Note that the constructor takes some time to complete the process of forming
and then inverting the operators.

# Constructors

- `FrequencyStreaming(Re, ϵ, Δx, xlimits, ylimits, b, [ddftype=Fields.Goza])`
where `b` is either a single `Body` or a `BodyList`.

# Use

The finished operator accepts x and y velocity amplitudes for each body and
returns the solution data structures for the first-order, second-order mean,
and Stokes drift solutions. E.g., for a single body `body`:

`soln1, soln2, solnsd = sys([1.0,0.0],body)`
"""
struct FrequencyStreaming{NX,NY,N} <: StreamingSystem{NX,NY,N}
    "Reynolds number (ΩL^2/ν)"
    Re::Float64

    "Amplitude parameter"
    ϵ::Float64

    "Grid metadata"
    grid::Fields.PhysicalGrid{2}

    # Operators
    "Laplacian operator"
    L::Fields.Laplacian{NX,NY}

    "Helmholtz operator"
    LH::Fields.Helmholtz{NX,NY}

    "Body coordinate data"
    X::VectorData{N,Float64}

    "Regularization and interpolation"
    reg::Regularize{N}
    Hmat::RegularizationMatrix
    Emat::InterpolationMatrix

    "Masks"
    inside::MaskType
    outside::MaskType
    dlayer::DoubleLayer

    "Saddle point systems"
    S₁::SaddleSystem
    S₂::SaddleSystem

end

function FrequencyStreaming(Re, ϵ, Δx,
                            xlimits::Tuple{Real,Real},ylimits::Tuple{Real,Real},
                            b::Union{Body,BodyList}; ddftype=Fields.Goza)

    X = VectorData(collect(b))

    # set up grid
    g = PhysicalGrid(xlimits,ylimits,Δx)
    NX, NY = size(g)
    N = length(X) ÷ 2

    # Basic data
    w = Nodes(Dual,(NX,NY),dtype=ComplexF64)
    f = VectorData(X,dtype=ComplexF64)

    L = plan_laplacian(NX,NY,with_inverse=true,dtype=ComplexF64)

    α = Re*Δx^2
    LH = plan_helmholtz(NX,NY,α,with_inverse=true)

    # Regularization and interpolation
    regop = Regularize(X,Δx,I0=origin(g),issymmetric=true,ddftype=ddftype)
    H, E = RegularizationMatrix(regop,VectorData(X,dtype=ComplexF64),Edges(Primal,(NX,NY),dtype=ComplexF64))

    # construct masks and double layer operator
    dlayer = DoubleLayer(b,regop,w)
    inside = Mask(dlayer)
    outside = ComplementaryMask(inside)


    # Set up operators
    #=
    First-order system
    =#
    B1₁ᵀ(f::VectorData{N,T}) where {N,T} = Curl()*(H*f)
    B1₂(w::Nodes{Dual,NX,NY,T}) where {NX,NY,T} = -(E*(Curl()*(L\w)))
    A1⁻¹(w::Nodes{Dual,NX,NY,T}) where {NX,NY,T} = LH\w

    #=
    Second-order mean system
    =#
    B2₂(w::Nodes{Dual,NX,NY,T}) where {NX,NY,T} = -(E*(Curl()*(L\outside(w))))
    A2⁻¹(w::Nodes{Dual,NX,NY,T}) where {NX,NY,T} = -(L\outside(w))

    # Set up saddle point systems

    S₁ = SaddleSystem((w,f),(A1⁻¹,B1₁ᵀ,B1₂),issymmetric=true,store=true)
    S₂ = SaddleSystem((w,f),(A2⁻¹,B1₁ᵀ,B2₂),issymmetric=true,store=true)

    return FrequencyStreaming{NX,NY,N}(Re,ϵ,g,L,LH,X,regop,H,E,inside,outside,dlayer,S₁,S₂)
end

function (sys::FrequencyStreaming{NX,NY,N})(U::Vector{Vector{T}},bl::BodyList) where {NX,NY,N,T<:Number}

    Ω = 1.0

    w1 = Nodes(Dual,size(sys),dtype=ComplexF64)
    w2 = Nodes(Dual,size(sys),dtype=ComplexF64)
    f1 = VectorData(sys.X,dtype=ComplexF64)
    f2 = VectorData(sys.X,dtype=ComplexF64)

    # first-order solution
    Ur₁ = zero(w1)
    Fr₁ = deepcopy(f1)
    for i in 1:length(bl)
        ui = view(Fr₁.u,bl,i)
        vi = view(Fr₁.v,bl,i)
        fill!(ui,U[i][1])
        fill!(vi,U[i][2])
    end
    rhs₁ = deepcopy((Ur₁,Fr₁))

    w1, f1 = sys.S₁\rhs₁

    ω₁ = vorticity(w1,sys)
    ψ₁ = streamfunction(w1,sys)
    u₁ = velocity(w1,sys)
    soln1 = FirstOrderComputational(sys.Re,sys.ϵ,Ω,sys.grid,ω₁,ψ₁,u₁)

    # construct drift flow
    udual = Nodes(Dual,w1)
    vdual = Nodes(Dual,w1)
    Fields.interpolate!(udual,u₁.u)
    Fields.interpolate!(vdual,u₁.v)
    sd = 0.5im/Ω*udual∘conj(vdual)/cellsize(sys)

    ψd = cellsize(sys)*sd
    ud = curl(sd)
    ωd = curl(ud)/cellsize(sys)
    udb = sys.Emat*ud

    sdsoln = SecondOrderMeanComputational(sys.Re,sys.ϵ,Ω,sys.grid,ωd,ψd,ud)

    # second-order mean solution
    rhs₂ = deepcopy((Re*Ūr₂(w1,sys),-udb))
    w2, f2 = sys.S₂\rhs₂

    ω₂ = vorticity(sys.outside(w2),sys)
    ψ₂ = streamfunction(sys.outside(w2),sys)
    u₂ = velocity(sys.outside(w2),sys)

    soln2 = SecondOrderMeanComputational(sys.Re,sys.ϵ,Ω,sys.grid,ω₂,ψ₂,u₂)

    return soln1, soln2, sdsoln

end

# For a single body
(sys::FrequencyStreaming{NX,NY,N})(U::Vector{T},body::Body) where {NX,NY,N,T<:Number} = sys([U],BodyList([body]))

#=
Types for holding the computational streaming solution
=#

abstract type StreamingComputational end

struct FirstOrderComputational{NX,NY} <: StreamingComputational
    Re :: Float64
    ϵ :: Float64
    Ω :: Float64
    g :: PhysicalGrid{2}
    W :: Nodes{Dual,NX,NY,ComplexF64}
    Ψ :: Nodes{Dual,NX,NY,ComplexF64}
    U :: Edges{Primal,NX,NY,ComplexF64}
end

struct SecondOrderMeanComputational{NX,NY} <: StreamingComputational
    Re :: Float64
    ϵ :: Float64
    Ω :: Float64
    g :: PhysicalGrid{2}
    W :: Nodes{Dual,NX,NY,ComplexF64}
    Ψ :: Nodes{Dual,NX,NY,ComplexF64}
    U :: Edges{Primal,NX,NY,ComplexF64}
end


#=
right-hand side functions
=#

function Ūr₂(w::Nodes{Dual,NX,NY,ComplexF64},sys::FrequencyStreaming{NX,NY}) where {NX,NY}

  Ww = Edges(Dual,w)
  Qq = Edges(Dual,w)
  Δx = cellsize(sys)

  Fields.interpolate!(Qq,curl(sys.L\conj(w))) # -velocity, on dual edges

  return 0.5*Δx*divergence(Qq∘Fields.interpolate!(Ww,w)) # -0.5*∇⋅(wu)

end
