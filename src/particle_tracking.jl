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