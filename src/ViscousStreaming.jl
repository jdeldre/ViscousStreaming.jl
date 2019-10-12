module ViscousStreaming

  using Reexport

  @reexport using ViscousFlow

  include("exact_onecylinder.jl")
  include("solver.jl")
  include("inertialparticles.jl")
  include("displacement.jl")
  include("averaging.jl")



end # module
