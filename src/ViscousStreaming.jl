module ViscousStreaming

  using Reexport
  using SpecialFunctions

  @reexport using ViscousFlow

  export params, StreamingParams, StreamingAnalytical, StreamingComputational

  abstract type OrderType end
  abstract type FirstOrder <: OrderType end
  abstract type SecondOrder <: OrderType end
  abstract type SecondOrderMean <: OrderType end

  abstract type FlowType end
  abstract type FluidFlow <: FlowType end
  abstract type ParticleFlow <: FlowType end

  """
      StreamingParams(ϵ,Re)

  Set the parameters for the streaming solution. The problem is scaled so
  that the radius of the cylinder is unity. Reynolds number is defined as Re = ΩR²/ν,
  where Ω is the angular frequency of the oscillatory motion, e.g. sin(Ωt).
  """
  struct StreamingParams
    ϵ :: Float64
    Re :: Float64
    Ω :: Float64
    γ² :: ComplexF64
    γ :: ComplexF64
    λ :: ComplexF64
    λ² :: ComplexF64
    H₀ :: ComplexF64
    C :: ComplexF64
  end

  function StreamingParams(ϵ::Number,Re::Number)
          γ² = im*Re
          γ = exp(im*π/4)*√Re
          λ = √2*γ
          λ² = 2*γ²
          H₀ = hankelh1(0,γ)
          C = hankelh1(2,γ)/H₀
          Ω = 1.0
          StreamingParams(ϵ,Re,Ω,γ²,γ,λ,λ²,H₀,C)
      end

  function Base.show(io::IO, p::StreamingParams) where {N}
          println(io, "Streaming flow parameters with Re = $(p.Re), ϵ = $(p.ϵ)")
  end

  """
    ComplexFunc(f)

  Provides a wrapper for a function expected to return complex values, for use in
  dispatch in automatic differentiation with `ForwardDiff`.
  """
  struct ComplexFunc{FT}
    fcn::FT
  end

  function Base.show(io::IO, f::ComplexFunc)
    println(io, "Complex function")
  end

  struct AsymptoticAnalytical{O <: OrderType}
      K  :: Integer
      p  :: StreamingParams
      Ψ :: ComplexFunc
      W :: ComplexFunc
      Ur :: ComplexFunc
      Uθ :: ComplexFunc
  end

  struct AsymptoticComputational{O <: OrderType, F <: FlowType, NX,NY}
      Re :: Float64
      ϵ :: Float64
      Ω :: Float64
      g :: PhysicalGrid{2}
      W :: Union{Nodes{Dual,NX,NY,ComplexF64},Nothing}
      Ψ :: Union{Nodes{Dual,NX,NY,ComplexF64},Nothing}
      U :: Edges{Primal,NX,NY,ComplexF64}
  end

  Base.size(::AsymptoticComputational{O,F,NX,NY}) where {O,F,NX,NY} = NX, NY

  abstract type StreamingSolution end

  struct StreamingAnalytical <: StreamingSolution
    p :: StreamingParams
    s1 :: AsymptoticAnalytical{FirstOrder}
    s2s :: AsymptoticAnalytical{SecondOrderMean}
    s2 :: AsymptoticAnalytical{SecondOrder}
  end

  struct StreamingComputational{F <: FlowType} <: StreamingSolution
    p :: StreamingParams
    g :: PhysicalGrid{2}
    s1 :: AsymptoticComputational{FirstOrder,F}
    s̄2 :: AsymptoticComputational{SecondOrderMean,F}
    sd :: AsymptoticComputational{SecondOrderMean,F}
    s2 :: Union{AsymptoticComputational{SecondOrder,F},Nothing}

  end

  params(s::T) where {T <: StreamingSolution} = s.p

  include("exact_onecylinder.jl")
  include("solver.jl")
  include("frequency_domain.jl")
  include("inertialparticles.jl")
  include("displacement.jl")
  include("averaging.jl")
  include("trajectories.jl")



end # module
