
import Base: *, +

using Interpolations
#using FastGaussQuadrature
#using LinearAlgebra
using SpecialFunctions
using ForwardDiff
using DiffRules

import ForwardDiff:value,partials,derivative,extract_derivative


export StreamingParams, FirstOrder, SecondOrder, SecondOrderMean, AnalyticalStreaming,
        uvelocity,vvelocity

"""
    StreamingParams(ϵ,Re)

Set the parameters for the streaming solution. The problem is scaled so
that the radius of the cylinder is unity. Reynolds number is defined as Re = ΩR²/ν,
where Ω is the angular frequency of the oscillatory motion, e.g. sin(Ωt).
"""
struct StreamingParams
    ϵ :: Float64
    Re :: Float64
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
        StreamingParams(ϵ,Re,γ²,γ,λ,λ²,H₀,C)
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

(f::ComplexFunc)(x) = f.fcn(x)

macro create_dual(f,nu,c,d,hankel)
    SF = :SpecialFunctions
    H = :($hankel)
    _f = Symbol("_",f)
    defs = quote
                $(_f)(x)  = $SF.$H($nu,$c*x)/$d
                $f = ComplexFunc($(_f))
           end
    return esc(defs)
end

macro extend_H(hankel)
    SF = :SpecialFunctions
    H = :($hankel)
    _, dH = DiffRules.diffrule(SF,H,:(ν),:(v))
    Hdef = quote
              function $SF.$H(ν,d::Complex{<:ForwardDiff.Dual{T}}) where {T}
                    dr, di = reim(d)
                    v = value(dr)+im*value(di)
                    return ForwardDiff.Dual{T}(real($SF.$H(ν,v)),real($dH)*partials(dr)-imag($dH)*partials(di)) +
                        im*ForwardDiff.Dual{T}(imag($SF.$H(ν,v)),imag($dH)*partials(dr)+real($dH)*partials(di))
              end
            end
    return esc(Hdef)
end

# Derivatives #

@inline function ForwardDiff.derivative(f::ComplexFunc{F}, x::R) where {F,R<:Real}
    T = typeof(ForwardDiff.Tag(f.fcn, R))
    return ForwardDiff.extract_derivative(T, real(f.fcn(ForwardDiff.Dual{T}(x, one(x))))) +
        im*ForwardDiff.extract_derivative(T, imag(f.fcn(ForwardDiff.Dual{T}(x, one(x)))))
end

"""
    D²(f::ComplexFunc,K::Int) -> ComplexFunc

Return the radial part of the Laplacian at `r` of a function having cylindrical form

\$ f(r)e^{iKθ} \$
"""
@inline function D²(f::ComplexFunc,K::Int)
      rdf = ComplexFunc(r -> r*derivative(f,r))
      drdf = ComplexFunc(r -> derivative(rdf,r))
      return ComplexFunc(r -> drdf(r)/r - K^2*f(r)/r^2)
end

"""
    curl(f::ComplexFunc,K::Int) -> Tuple{ComplexFunc,ComplexFunc}

Return the radial part of the cylindrical components of the curl at `r` of a
function having cylindrical form

\$ f(r)e^{iKθ} \$
"""
@inline function curl(f::ComplexFunc,K::Int)
  return ComplexFunc(r -> K*f(r)/r),ComplexFunc(r -> -derivative(f,r))
end

# Integrals #
"""
    Integral(f,a<:Real,b<:Real[,fake_infinity=1e5],[,length=10000])

Set up an integral

\$ \\int_a^r f(\\tau)d\\tau \$

for a < r < b
If `b` is set to `Inf`, then this sets up the integral

\$ \\int_r^\\infty f(\\tau)d\\tau \$

where a < r < ∞
The optional argument `length` sets the size of the interpolation table, and
`fake_infinity` sets the maximum value at which to truncate the limit at
infinity.
"""
struct Integral{FT,V,IL}
    fcn::FT
    a::Float64
    b::Float64
    fake_infinity::Float64
    table::Interpolations.AbstractInterpolation
end

function Integral(fcn,a,b;fake_infinity=1e6,length=10000)
    V = typeof(fcn(1))
    v = zeros(V,length)
    τ = range(0,stop=1,length=length)
    if b == Inf
        # Integration variable τ = c/x - c/fake_infinity, where c is
        # such that τ goes from 1 to 0 when x goes from a to fake_infinity.
        lim_diff = 1/a-1/fake_infinity
        c = 1/lim_diff
        xi = c/(0+c/fake_infinity)
        fcn_old = fcn(xi)*xi^2/c
        for (i,τi) in enumerate(τ)
          if i > 1
            xi = c/(τi+c/fake_infinity)
            fcn_new = fcn(xi)*xi^2/c
            v[i] += v[i-1] + 0.5*step(τ)*(fcn_new+fcn_old)
            fcn_old = fcn_new
          end
        end
        table = LinearInterpolation(τ, v, extrapolation_bc = Flat())
        Integral{typeof(fcn),V,true}(fcn,a,b,fake_infinity,table)
    else
        # Integration variable τ = c(x - a), where c is such
        # that τ goes from 0 to 1 when x goes from a to b
        lim_diff = b-a
        c = 1/lim_diff
        fcn_old = fcn(a)
        for (i,τi) in enumerate(τ)
          if i > 1
            xi = a + τi*lim_diff
            fcn_new = fcn(xi)
            v[i] += v[i-1] + 0.5*step(τ)*lim_diff*(fcn_new+fcn_old)
            fcn_old = fcn_new
          end
        end
        table = LinearInterpolation(τ, v, extrapolation_bc = Flat())
        Integral{typeof(fcn),V,false}(fcn,a,b,fake_infinity,table)
    end
end

function Base.show(io::IO, I::Integral{FT,V,IL}) where {FT,V,IL}
    if IL
      println(io, "Integral from $(I.a) < r to $(I.b)")
    else
      println(io, "Integral from $(I.a) to r < $(I.b)")
    end
end

ComplexIntegral(fcn::ComplexFunc,a,b;ops...) = ComplexFunc(Integral(fcn,a,b;ops...))

ComplexIntegral(fcn::Function,a,b;ops...) = ComplexFunc(Integral(ComplexFunc(fcn),a,b;ops...))

function (I::Integral{FT,V,false})(r) where {FT,V<:Number}
  return r > I.a ? I.table((r-I.a)/(I.b-I.a)) : zero(V)
end

function (I::Integral{FT,V,true})(r) where {FT,V<:Number}
  return I.table((1/r-1/I.fake_infinity)/(1/I.a-1/I.fake_infinity))
end

#(I::Integral{FT,V,true} where {FT,V<:Number})(d::ForwardDiff.Dual{T}) where {T} = ForwardDiff.Dual{T}(I(value(d)),-I.fcn(value(d))*partials(d))

#(I::Integral{FT,V,false} where {FT,V<:Number})(d::ForwardDiff.Dual{T}) where {T} = ForwardDiff.Dual{T}(I(value(d)),I.fcn(value(d))*partials(d))

# Now define how they behave with ForwardDiff.Dual types. The following defines the automatic differentiation:
function (I::Integral{FT,V,true} where {FT<:ComplexFunc,V<:Number})(d::ForwardDiff.Dual{T}) where {T}
          return ForwardDiff.Dual{T}(real(I(value(d))),-real(I.fcn(value(d)))*partials(d)) +
       im*ForwardDiff.Dual{T}(imag(I(value(d))),-imag(I.fcn(value(d)))*partials(d))
end

(I::Integral{FT,V,false} where {FT<:ComplexFunc,V<:Number})(d::ForwardDiff.Dual{T}) where {T} =
         ForwardDiff.Dual{T}(real(I(value(d))),real(I.fcn(value(d)))*partials(d)) +
      im*ForwardDiff.Dual{T}(imag(I(value(d))),imag(I.fcn(value(d)))*partials(d))

# Set up functions #

@extend_H(hankelh1)
@extend_H(hankelh2)




abstract type Order end
abstract type First <: Order end
abstract type Second <: Order end

struct ComplexAmplitude{FT,K}
    f :: FT
end

ComplexAmplitude(f,K) = ComplexAmplitude{typeof(f),K == 1 ? First : Second}(f)

function (A::ComplexAmplitude{FT,Second})(x,y) where {FT}
    r2 = x^2+y^2
    r = sqrt(r2)
    return 2*real(A.f(r))*x*y/r2
end

#### Construct the first and second order solutions

struct FirstOrder
    K  :: Integer
    p  :: StreamingParams
    Ψ₁ :: ComplexFunc
    W₁ :: ComplexFunc
    Ur₁ :: ComplexFunc
    Uθ₁ :: ComplexFunc
end

function FirstOrder(p::StreamingParams)

    @create_dual(Y,1,p.γ,p.H₀,hankelh1)

    K = 1
    Ψ₁ = ComplexFunc(r -> -p.C/r + 2Y(r)/p.γ)
    W₁ = D²(Ψ₁,K)  # note that this is actually the negative of the vorticity. We will account for this when we evaluate it.
    Ur₁, Uθ₁ = curl(Ψ₁,K)

    # for verifying the solution
    LW₁ = D²(W₁,K);
    resid1 = ComplexFunc(r -> LW₁(r)+im*p.Re*W₁(r))
    println("Maximum residual on W₁ = ",maximum(abs.(resid1.(range(1,5,length=10)))))

    # for verifying boundary conditions
    dΨ₁ = ComplexFunc(r -> derivative(Ψ₁,r))
    bcresid1 = Ψ₁(1) - 1
    bcresid2 = dΨ₁(1) - 1

    println("BC residual on Ψ₁(1) = ",abs(bcresid1))
    println("BC residual on dΨ₁(1) = ",abs(bcresid2))

    return FirstOrder(K,p,Ψ₁,W₁,Ur₁,Uθ₁)

end

function Base.show(io::IO, s::FirstOrder)
        println(io, "First-order analytical streaming flow solution for")
        println(io, "single cylinder with Re = $(s.p.Re), ϵ = $(s.p.ϵ)")
end

function vorticity(x,y,t,s::FirstOrder)
  r = sqrt(x^2+y^2)
  return real(-s.W₁(r)*y/r*exp.(-im*t))
end
function uvelocity(x,y,t,s::FirstOrder)
    r = sqrt(x^2+y^2)
    coseval = x/r
    sineval = y/r
    return real.((s.Ur₁(r)*coseval^2-s.Uθ₁(r)*sineval^2)*exp.(-im*t))
end
function vvelocity(x,y,t,s::FirstOrder)
    r = sqrt(x^2+y^2)
    coseval = x/r
    sineval = y/r
    return real.((s.Ur₁(r)+s.Uθ₁(r))*coseval*sineval*exp.(-im*t))
end
function streamfunction(x,y,t,s::FirstOrder)
    r = sqrt(x^2+y^2)
    return real(s.Ψ₁(r)*y/r*exp.(-im*t))
end

# second order mean

struct SecondOrderMean
    K  :: Integer
    p  :: StreamingParams
    Ψ₂ :: ComplexFunc
    W₂ :: ComplexFunc
    Ur₂ :: ComplexFunc
    Uθ₂ :: ComplexFunc
end

function SecondOrderMean(p::StreamingParams;n1inf=100000,n120=400000)

  @create_dual(X,0,p.γ,p.H₀,hankelh1)
  @create_dual(Y,1,p.γ,p.H₀,hankelh1)
  @create_dual(Z,2,p.γ,p.H₀,hankelh1)


  K = 2
  fakefact = 1
  #f₀ = ComplexFunc(r -> -0.5*p.γ²*p.Re*(0.5*(p.C*conj(X(r))-conj(p.C)*X(r))/r^2 + X(r)*conj(Z(r)) - conj(X(r))*Z(r)))
  f₀ = ComplexFunc(r -> -p.γ²*p.Re*(0.5*p.C*conj(X(r))/r^2 + X(r)*conj(Z(r))))

  f̃₀ = ComplexFunc(r -> f₀(r) - 0.5*p.γ²*p.Re*(-0.5*conj(Z(r))+0.5*Z(r)))
  I⁻¹ = ComplexIntegral(r->f₀(r)/r,1,Inf,length=n1inf)
  I¹ = ComplexIntegral(r->f₀(r)*r,1,Inf,length=n1inf)
  I³ = ComplexIntegral(r->f₀(r)*r^3,1,20,length=n120)
  I⁵ = ComplexIntegral(r->f₀(r)*r^5,1,20,length=n120)
  Ψs₂ = ComplexFunc(r -> -r^4/48*I⁻¹(r) + r^2/16*I¹(r) + I³(r)/16 + I⁻¹(1)/16 - I¹(1)/8 - fakefact*0.25im*p.γ*Y(1) +
  1/r^2*(-I⁵(r)/48-I⁻¹(1)/24+I¹(1)/16 + fakefact*0.25im*p.γ*Y(1)))
  Ws₂ = D²(Ψs₂,K)
  Usr₂, Usθ₂ = curl(Ψs₂,K)

  # for verifying the solution
  LWs₂ = D²(Ws₂,K)
  resids = ComplexFunc(r -> real(LWs₂(r)-f₀(r)))
  println("Maximum residual on Ws₂ = ",maximum(abs.(resids.(range(1,5,length=10)))))

  # for verifying boundary conditions
  dΨs₂ = ComplexFunc(r -> derivative(Ψs₂,r))
  Ψ₁ = ComplexFunc(r -> -p.C/r + 2Y(r)/p.γ)
  W₁ = D²(Ψ₁,K)
  bcresids1 = Ψs₂(1)
  bcresids2 = real(dΨs₂(1) - 0.25im*W₁(1))

  println("BC residual on Ψs₂(1) = ",abs(bcresids1))
  println("BC residual on dΨs₂(1) = ",abs(bcresids2))

  return SecondOrderMean(K,p,Ψs₂,Ws₂,Usr₂, Usθ₂)
end

function Base.show(io::IO, s::SecondOrderMean)
        println(io, "Second-order mean part of analytical streaming flow solution for")
        println(io, "single cylinder with Re = $(s.p.Re), ϵ = $(s.p.ϵ)")
end

function vorticity(x,y,s::SecondOrderMean)
    r = sqrt(x^2+y^2)
    sin2eval = 2*x*y/r^2
    return real(-s.W₂(r))*sin2eval
end
function uvelocity(x,y,s::SecondOrderMean)
    r = sqrt(x^2+y^2)
    coseval = x/r
    sineval = y/r
    cos2eval = coseval^2-sineval^2
    sin2eval = 2*coseval*sineval
    ur = real.(s.Ur₂(r))*cos2eval
    uθ = real.(s.Uθ₂(r))*sin2eval
    return ur*coseval .- uθ*sineval
end
function vvelocity(x,y,s::SecondOrderMean)
    r = sqrt(x^2+y^2)
    coseval = x/r
    sineval = y/r
    cos2eval = coseval^2-sineval^2
    sin2eval = 2*coseval*sineval
    ur = real.(s.Ur₂(r))*cos2eval
    uθ = real.(s.Uθ₂(r))*sin2eval
    return ur*sineval .+ uθ*coseval
end
function streamfunction(x,y,s::SecondOrderMean)
    r = sqrt(x^2+y^2)
    coseval = x/r
    sineval = y/r
    sin2eval = 2*coseval*sineval
    return real(s.Ψ₂(r))*sin2eval
end


# second order unsteady
struct SecondOrder
    K  :: Integer
    p  :: StreamingParams
    Ψ₂ :: ComplexFunc
    W₂ :: ComplexFunc
    Ur₂ :: ComplexFunc
    Uθ₂ :: ComplexFunc
end

function SecondOrder(p::StreamingParams;n1inf=100000,n120=400000)

  @create_dual(X,0,p.γ,p.H₀,hankelh1)
  @create_dual(Y,1,p.γ,p.H₀,hankelh1)
  @create_dual(Z,2,p.γ,p.H₀,hankelh1)

  @create_dual(H11,1,p.λ,1,hankelh1)
  @create_dual(H12,1,p.λ,1,hankelh2)
  @create_dual(H21,2,p.λ,1,hankelh1)
  @create_dual(H22,2,p.λ,1,hankelh2)

  K = 2
  fakefact = 1
  g₀ = ComplexFunc(r -> 0.5*p.γ²*p.Re*p.C*X(r)/r^2)

  g̃₀ = ComplexFunc(r -> g₀(r) - 0.5*p.γ²*p.Re*Z(r))


  Kλ = ComplexFunc(r -> H11(1)*H22(r) - H12(1)*H21(r))

  IKgr = ComplexIntegral(r -> r*Kλ(r)*g₀(r),1,20,length=n120)
  IH21gr = ComplexIntegral(r -> r*H21(r)*g₀(r),1,Inf,length=n1inf)
  Igr⁻¹ = ComplexIntegral(r -> g₀(r)/r,1,Inf,length=n1inf)
  Igr³ = ComplexIntegral(r -> g₀(r)*r^3,1,20,length=n120)

  Ig¹ = ComplexFunc(r -> 0.25im*π/(p.λ²*H11(1))*IKgr(r)*H21(r))
  Ig² = ComplexFunc(r -> 0.25im*π/(p.λ²*H11(1))*IH21gr(r)*Kλ(r))
  Ig³ = ComplexFunc(r -> 1/(p.λ²*p.λ*H11(1))*((H21(r)-H21(1)/r^2)*Igr⁻¹(1)+IH21gr(1)/r^2))
  Ig⁴ = ComplexFunc(r -> -0.25/p.λ²*(Igr⁻¹(r)*r^2-Igr⁻¹(1)/r^2+Igr³(r)/r^2))
  Ψ₂ = ComplexFunc(r -> Ig¹(r) + Ig²(r) + Ig³(r) + Ig⁴(r) + fakefact*0.5im/sqrt(2)*Y(1)/H11(1)*(H21(r)-H21(1)/r^2))

  Ψ̃₂ = ComplexFunc(r -> Ψ₂(r)+ 0.5im*(-p.C/r^2 + Z(r))) # cylinder-fixed reference frame... not used
  W₂ = D²(Ψ₂,K)
  Ur₂, Uθ₂ = curl(Ψ₂,K);

  # for verifying the solution
  LW₂ = D²(W₂,K);
  resid = ComplexFunc(r -> LW₂(r)+2im*p.Re*W₂(r)-g₀(r))
  println("Maximum residual on W₂ = ",maximum(abs.(resid.(range(1,5,length=10)))))

  # for verifying boundary conditions
  dΨ₂ = ComplexFunc(r -> derivative(Ψ₂,r))
  bcresid1 = Ψ₂(1)
  bcresid2 = dΨ₂(1) - 0.5im*p.γ*Y(1)

  println("BC residual on Ψ₂(1) = ",abs(bcresid1))
  println("BC residual on dΨ₂(1) = ",abs(bcresid2))

  return SecondOrder(K,p,Ψ₂,W₂,Ur₂, Uθ₂)
end

function Base.show(io::IO, s::SecondOrder)
        println(io, "Second-order oscillatory part of analytical streaming flow solution for")
        println(io, "single cylinder with Re = $(s.p.Re), ϵ = $(s.p.ϵ)")
end

function vorticity(x,y,t,s::SecondOrder)
    r = sqrt(x^2+y^2)
    sin2eval = 2*x*y/r^2
    return real(-s.W₂(r)*exp.(-2im*t))*sin2eval
end
function uvelocity(x,y,t,s::SecondOrder)
    r = sqrt(x^2+y^2)
    coseval = x/r
    sineval = y/r
    cos2eval = coseval^2-sineval^2
    sin2eval = 2*coseval*sineval
    ur = real.(s.Ur₂(r)*exp.(-2im*t))*cos2eval
    uθ = real.(s.Uθ₂(r)*exp.(-2im*t))*sin2eval
    return ur*coseval .- uθ*sineval
end
function vvelocity(x,y,t,s::SecondOrder)
    r = sqrt(x^2+y^2)
    coseval = x/r
    sineval = y/r
    cos2eval = coseval^2-sineval^2
    sin2eval = 2*coseval*sineval
    ur = real.(s.Ur₂(r)*exp.(-2im*t))*cos2eval
    uθ = real.(s.Uθ₂(r)*exp.(-2im*t))*sin2eval
    return ur*sineval .+ uθ*coseval
end
function streamfunction(x,y,t,s::SecondOrder)
    r = sqrt(x^2+y^2)
    coseval = x/r
    sineval = y/r
    sin2eval = 2*coseval*sineval
    return real(s.Ψ₂(r)*exp.(-2im*t))*sin2eval
end

### all together
struct AnalyticalStreaming
  p :: StreamingParams
  s1 :: FirstOrder
  s2s :: SecondOrderMean
  s2 :: SecondOrder
end

AnalyticalStreaming(p) = AnalyticalStreaming(p,FirstOrder(p),SecondOrderMean(p),SecondOrder(p))

function Base.show(io::IO, s::AnalyticalStreaming)
        println(io, "Analytical streaming flow solution for")
        println(io, "single cylinder with Re = $(s.p.Re), ϵ = $(s.p.ϵ)")
end


vorticity(x,y,t,s::AnalyticalStreaming) =
      s.p.ϵ*vorticity(x,y,t,s.s1) + s.p.ϵ^2*(vorticity(x,y,s.s2s)+vorticity(x,y,t,s.s2))

uvelocity(x,y,t,s::AnalyticalStreaming) =
      s.p.ϵ*uvelocity(x,y,t,s.s1) + s.p.ϵ^2*(uvelocity(x,y,s.s2s)+uvelocity(x,y,t,s.s2))

vvelocity(x,y,t,s::AnalyticalStreaming) =
      s.p.ϵ*vvelocity(x,y,t,s.s1) + s.p.ϵ^2*(vvelocity(x,y,s.s2s)+vvelocity(x,y,t,s.s2))

streamfunction(x,y,t,s::AnalyticalStreaming) =
      s.p.ϵ*streamfunction(x,y,t,s.s1) + s.p.ϵ^2*(streamfunction(x,y,s.s2s)+streamfunction(x,y,t,s.s2))
