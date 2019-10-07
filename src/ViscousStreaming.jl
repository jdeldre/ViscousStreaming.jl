module ViscousStreaming

  using Reexport

  @reexport using ViscousFlow

  include("exact_onecylinder.jl")
  include("solver.jl")


end # module
