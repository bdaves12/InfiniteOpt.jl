# include("C:/Users/puls446/.julia/dev/InfiniteOpt/examples/test.jl")

using Revise, JuMP, InfiniteOpt, Distributions

space_sets = [IntervalSet(-1, 1) for i = 1:6]

dist = Normal()
samples = rand(dist, 2)

m = InfiniteModel()

@infinite_parameter(m, 0 <= t <= 6, supports = [0, 3])
x = @infinite_parameter(m, [i = 1:6], set = space_sets[i], base_name = "x", independent = true)
@infinite_parameter(m, xi in dist, supports = samples)

add_supports(t, 6)

for i = 1:length(x)
    set_supports(x[i], Vector(-1:1:1))
end

@infinite_variable(m, j[1:2](t)) # unused variable
@infinite_variable(m, g(t), Int)
@infinite_variable(m, 0 <= T(t, x))
@infinite_variable(m, -10 <= w(xi) <= 10)
q = @infinite_variable(m, parameter_refs = (t, x, xi))

@point_variable(m, T(6, -ones(Int64,6)), Tf == 1)

@global_variable(m, 0 <= z <= 42)

tdata = DiscreteMeasureData(t, ones(3) * 0.5, [0, 2, 6])
rdata = DiscreteMeasureData(xi, ones(length(samples)), supports(xi), name = "E")
xdata = DiscreteMeasureData(x, ones(length(supports(x))) * 2, supports(x), name = "S")

@objective(m, Min, measure(measure(T^2, xdata) - 3g, tdata) + measure(w, rdata) + 3z -2)

@constraint(m, Tf + z <= 10)
@constraint(m, T == measure(T + 3, tdata), parameter_bounds = Dict(t => IntervalSet(0, 2)))
@constraint(m, w <= g * xi)
expr = @expression(m, measure(2w + measure(measure(q + T, tdata), xdata), rdata))
@constraint(m, expr == 0)

print(m)
println("")

#mt = TranscriptionModel(m);
