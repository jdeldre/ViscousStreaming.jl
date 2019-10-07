module ViscousStreaming

  using Reexport

  @reexport using ViscousFlow

  include("fields.jl")
  include("exact_onecylinder.jl")
  include("solver.jl")


end # module
