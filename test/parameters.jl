@testset "Base Extensions" begin
    # Define test model and pref
    m = InfiniteModel()
    m2 = InfiniteModel()
    pref = InfOptVariableRef(m, 1, Parameter)
    # test variable compare
    @testset "(==)" begin
        @test pref == pref
        @test pref == InfOptVariableRef(m, 1, Parameter)
        @test !(pref == InfOptVariableRef(m, 2, Parameter))
        @test !(pref == InfOptVariableRef(m2, 1, Parameter))
        @test !(pref != InfOptVariableRef(m, 1, Parameter))
    end
    # test copy(v)
    @testset "copy(v)" begin
        @test copy(pref) == pref
    end
    # test copy(v, m)
    @testset "copy(v, m)" begin
        @test copy(pref, m2) == InfOptVariableRef(m2, 1, Parameter)
    end
    # test broadcastable
    @testset "broadcastable" begin
        @test isa(Base.broadcastable(pref), Base.RefValue{InfOptVariableRef})
    end
end

# Test macro methods
@testset "Macro Helpers" begin
    @testset "Symbol Methods" begin
        @test Parameter == :Parameter
        @test InfiniteOpt._is_set_keyword(:(lower_bound = 0))
    end
    # test ParameterInfoExpr datatype
    @testset "_ParameterInfoExpr" begin
        @test InfiniteOpt._ParameterInfoExpr isa DataType
        @test InfiniteOpt._ParameterInfoExpr(ones(Bool, 8)...).has_lb
        @test !InfiniteOpt._ParameterInfoExpr().has_lb
    end
    # test JuMP._set_lower_bound_or_error
    @testset "JuMP._set_lower_bound_or_error" begin
        info = InfiniteOpt._ParameterInfoExpr()
        # test normal operation
        @test isa(JuMP._set_lower_bound_or_error(error, info, 0), Nothing)
        @test info.has_lb && info.lower_bound == 0
        # test double/lack of input errors
        @test_throws ErrorException JuMP._set_lower_bound_or_error(error,
                                                                   info, 0)
        info.has_lb = false; info.has_dist = true
        @test_throws ErrorException JuMP._set_lower_bound_or_error(error,
                                                                   info, 0)
        info.has_dist = false; info.has_set = true
        @test_throws ErrorException JuMP._set_lower_bound_or_error(error,
                                                                   info, 0)
    end
    # test JuMP._set_upper_bound_or_error
    @testset "JuMP._set_upper_bound_or_error" begin
        info = InfiniteOpt._ParameterInfoExpr()
        # test normal operation
        @test isa(JuMP._set_upper_bound_or_error(error, info, 0), Nothing)
        @test info.has_ub && info.upper_bound == 0
        # test double/lack of input errors
        @test_throws ErrorException JuMP._set_upper_bound_or_error(error,
                                                                   info, 0)
        info.has_ub = false; info.has_dist = true
        @test_throws ErrorException JuMP._set_upper_bound_or_error(error,
                                                                   info, 0)
        info.has_dist = false; info.has_set = true
        @test_throws ErrorException JuMP._set_upper_bound_or_error(error,
                                                                   info, 0)
    end
    # _dist_or_error
    @testset "_dist_or_error" begin
        info = InfiniteOpt._ParameterInfoExpr()
        # test normal operation
        @test isa(InfiniteOpt._dist_or_error(error, info, 0), Nothing)
        @test info.has_dist && info.distribution == 0
        # test double/lack of input errors
        @test_throws ErrorException InfiniteOpt._dist_or_error(error, info, 0)
        info.has_dist = false; info.has_lb = true
        @test_throws ErrorException InfiniteOpt._dist_or_error(error, info, 0)
        info.has_lb = false; info.has_set = true
        @test_throws ErrorException InfiniteOpt._dist_or_error(error, info, 0)
    end
    # _set_or_error
    @testset "_set_or_error" begin
        info = InfiniteOpt._ParameterInfoExpr()
        # test normal operation
        @test isa(InfiniteOpt._set_or_error(error, info, 0), Nothing)
        @test info.has_set && info.set == 0
        # test double/lack of input errors
        @test_throws ErrorException InfiniteOpt._set_or_error(error, info, 0)
        info.has_set = false; info.has_lb = true
        @test_throws ErrorException InfiniteOpt._set_or_error(error, info, 0)
        info.has_lb = false; info.has_dist = true
        @test_throws ErrorException InfiniteOpt._set_or_error(error, info, 0)
    end
    # _constructor_set
    @testset "InfiniteOpt._constructor_set" begin
        info = InfiniteOpt._ParameterInfoExpr()
        @test_throws ErrorException InfiniteOpt._constructor_set(error, info)
        info.has_lb = true; info.lower_bound = 0
        @test_throws ErrorException InfiniteOpt._constructor_set(error, info)
        info.has_ub = true; info.upper_bound = 1
        check = :(isa($(info.lower_bound), Number))
        expected = :($(check) ? IntervalSet($(info.lower_bound), $(info.upper_bound)) : error("Bounds must be a number."))
        @test InfiniteOpt._constructor_set(error, info) == expected
        info = InfiniteOpt._ParameterInfoExpr(distribution = Normal())
        check = :(isa($(info.distribution), Distributions.NonMatrixDistribution))
        expected = :($(check) ? DistributionSet($(info.distribution)) : error("Distribution must be a subtype of Distributions.NonMatrixDistribution."))
        @test InfiniteOpt._constructor_set(error, info) == expected
        info = InfiniteOpt._ParameterInfoExpr(set = IntervalSet(0, 1))
        check = :(isa($(info.set), AbstractInfiniteSet))
        expected = :($(check) ? $(info.set) : error("Set must be a subtype of AbstractInfiniteSet."))
        @test InfiniteOpt._constructor_set(error, info) == expected
    end
    # _parse_one_operator_parameter
    @testset "_parse_one_operator_parameter" begin
        info = InfiniteOpt._ParameterInfoExpr()
        @test isa(InfiniteOpt._parse_one_operator_parameter(error, info,
                                                          Val(:<=), 0), Nothing)
        @test info.has_ub && info.upper_bound == 0
        @test isa(InfiniteOpt._parse_one_operator_parameter(error, info,
                                                          Val(:>=), 0), Nothing)
        @test info.has_lb && info.lower_bound == 0
        info = InfiniteOpt._ParameterInfoExpr()
        @test isa(InfiniteOpt._parse_one_operator_parameter(error, info,
                                                          Val(:in), 0), Nothing)
        @test info.has_dist && info.distribution == 0
        @test_throws ErrorException InfiniteOpt._parse_one_operator_parameter(error, info,
                                                                              Val(:d), 0)
    end
    # _parse_ternary_parameter
    @testset "_parse_ternary_parameter" begin
        info = InfiniteOpt._ParameterInfoExpr()
        @test isa(InfiniteOpt._parse_ternary_parameter(error, info, Val(:<=), 0,
                                                       Val(:<=), 0), Nothing)
        @test info.has_ub && info.upper_bound == 0
        @test info.has_lb && info.lower_bound == 0
        info = InfiniteOpt._ParameterInfoExpr()
        @test isa(InfiniteOpt._parse_ternary_parameter(error, info, Val(:>=), 0,
                                                       Val(:>=), 0), Nothing)
        @test info.has_ub && info.upper_bound == 0
        @test info.has_lb && info.lower_bound == 0
        @test_throws ErrorException InfiniteOpt._parse_ternary_parameter(error,
                                                 info, Val(:<=), 0, Val(:>=), 0)
    end
    # _parse_parameter
    @testset "_parse_parameter" begin
        info = InfiniteOpt._ParameterInfoExpr()
        @test InfiniteOpt._parse_parameter(error, info, :<=, :x, 0) == :x
        @test info.has_ub && info.upper_bound == 0
        @test InfiniteOpt._parse_parameter(error, info, :<=, 0, :x) == :x
        @test info.has_lb && info.lower_bound == 0
        info = InfiniteOpt._ParameterInfoExpr()
        @test InfiniteOpt._parse_parameter(error, info, 0, :<=, :x, :<=, 0) == :x
        @test info.has_ub && info.upper_bound == 0
        @test info.has_lb && info.lower_bound == 0
    end
end

# Test precursor functions needed for add_parameter
@testset "Basic Reference Queries" begin
    m = InfiniteModel()
    pref = InfOptVariableRef(m, 1, Parameter)
    # JuMP.index
    @testset "JuMP.index" begin
        @test JuMP.index(pref) == 1
    end
    # JuMP.owner_model
    @testset "JuMP.owner_model" begin
        @test owner_model(pref) == m
    end
end

# Test name methods
@testset "Name" begin
    m = InfiniteModel()
    param = InfOptParameter(IntervalSet(0, 1), Number[], false)
    m.params[1] = param
    m.param_to_name[1] = "test"
    m.name_to_param = nothing
    pref = InfOptVariableRef(m, 1, Parameter)
    # JuMP.name
    @testset "JuMP.name" begin
        @test name(pref) == "test"
    end
    # JuMP.set_name
    @testset "JuMP.set_name" begin
        @test isa(set_name(pref, "new"), Nothing)
        @test name(pref) == "new"
    end
    # parameter_by_name
    @testset "parameter_by_name" begin
        @test parameter_by_name(m, "new") == pref
        @test isa(parameter_by_name(m, "test2"), Nothing)
        m.params[2] = param
        m.param_to_name[2] = "new"
        m.name_to_param = nothing
        @test_throws ErrorException parameter_by_name(m, "new")
    end
end

# Test parameter definition methods
@testset "Definition" begin
    # _check_supports_in_bounds
    @testset "_check_supports_in_bounds" begin
        set = IntervalSet(0, 1)
        @test isa(InfiniteOpt._check_supports_in_bounds(error, 0, set), Nothing)
        @test_throws ErrorException InfiniteOpt._check_supports_in_bounds(error,
                                                                        -1, set)
        @test_throws ErrorException InfiniteOpt._check_supports_in_bounds(error,
                                                                         2, set)
        set = DistributionSet(Uniform())
        @test isa(InfiniteOpt._check_supports_in_bounds(error, 0, set), Nothing)
        @test_throws ErrorException InfiniteOpt._check_supports_in_bounds(error,
                                                                        -1, set)
        @test_throws ErrorException InfiniteOpt._check_supports_in_bounds(error,
                                                                         2, set)
    end
    # build_parameter
    @testset "build_parameter" begin
        set = DistributionSet(Multinomial(3, 2))
        supps = Vector(0:1)
        expected = InfOptParameter(set, supps, false)
        @test build_parameter(error, set, 2, supports = supps) == expected
        expected = InfOptParameter(set, Number[], true)
        @test build_parameter(error, set, 2, independent = true) == expected
        @test_throws ErrorException build_parameter(error, set, 2, bob = 42)
        @test_throws ErrorException build_parameter(error, set, 1)
        warn = "Support points are not unique, eliminating redundant points."
        @test_logs (:warn, warn) build_parameter(error, set, 2,
                                                 supports = ones(3))
    end
    # add_parameter
    @testset "add_parameter" begin
        m = InfiniteModel()
        param = InfOptParameter(IntervalSet(0, 1), Number[], false)
        expected = InfOptVariableRef(m, 1, Parameter)
        @test add_parameter(m, param) == expected
        @test m.params[1] isa InfOptParameter
        @test m.param_to_group_id[1] == 1
    end
end

# Test the parameter macro
@testset "Macro" begin
    m = InfiniteModel()
    # single parameter
    @testset "Single" begin
        pref = InfOptVariableRef(m, 1, Parameter)
        @test @infinite_parameter(m, 0 <= a <= 1) == pref
        @test m.params[1].set == IntervalSet(0, 1)
        @test name(pref) == "a"
        pref = InfOptVariableRef(m, 2, Parameter)
        @test @infinite_parameter(m, b in Normal(), supports = [1; 2]) == pref
        @test m.params[2].set == DistributionSet(Normal())
        @test m.params[2].supports == [1, 2]
        pref = InfOptVariableRef(m, 3, Parameter)
        @test @infinite_parameter(m, c, set = IntervalSet(0, 1)) == pref
        pref = InfOptVariableRef(m, 4, Parameter)
        @test @infinite_parameter(m, set = IntervalSet(0, 1),
                                  base_name = "d") == pref
        @test name(pref) == "d"
        pref = InfOptVariableRef(m, 5, Parameter)
        @test @infinite_parameter(m, set = IntervalSet(0, 1)) == pref
        @test name(pref) == ""
    end
    # multiple parameters
    @testset "Array" begin
        prefs = [InfOptVariableRef(m, 6, Parameter),
                 InfOptVariableRef(m, 7, Parameter)]
        @test @infinite_parameter(m, 0 <= e[1:2] <= 1) == prefs
        @test m.params[6].set == IntervalSet(0, 1)
        @test m.params[7].set == IntervalSet(0, 1)
        prefs = [InfOptVariableRef(m, 8, Parameter),
                 InfOptVariableRef(m, 9, Parameter)]
        @test @infinite_parameter(m, [1:2], set = IntervalSet(0, 1)) == prefs
        @test m.params[8].set == IntervalSet(0, 1)
        @test m.params[9].set == IntervalSet(0, 1)
        prefs = [InfOptVariableRef(m, 10, Parameter),
                 InfOptVariableRef(m, 11, Parameter)]
        sets = [IntervalSet(0, 1), IntervalSet(-1, 2)]
        @test @infinite_parameter(m, f[i = 1:2], set = sets[i]) == prefs
        @test m.params[10].set == IntervalSet(0, 1)
        @test m.params[11].set == IntervalSet(-1, 2)
        prefs = [InfOptVariableRef(m, 12, Parameter),
                 InfOptVariableRef(m, 13, Parameter)]
        @test @infinite_parameter(m, [i = 1:2], set = sets[i]) == prefs
        @test m.params[12].set == IntervalSet(0, 1)
        @test m.params[13].set == IntervalSet(-1, 2)
        prefs = [InfOptVariableRef(m, 14, Parameter),
                 InfOptVariableRef(m, 15, Parameter)]
        @test @infinite_parameter(m, [0, -1][i] <= g[i = 1:2] <= [1, 2][i]) == prefs
        @test m.params[14].set == IntervalSet(0, 1)
        @test m.params[15].set == IntervalSet(-1, 2)
        prefs = [InfOptVariableRef(m, 16, Parameter),
                 InfOptVariableRef(m, 17, Parameter)]
        @test @infinite_parameter(m, 0 <= h[1:2] <= 1,
                                  independent = true) == prefs
        @test m.params[16].independent
        @test m.params[17].independent
        prefs = [InfOptVariableRef(m, 18, Parameter),
                 InfOptVariableRef(m, 19, Parameter)]
        prefs = convert(JuMP.Containers.SparseAxisArray, prefs)
        @test @infinite_parameter(m, 0 <= i[1:2] <= 1,
                                  container = SparseAxisArray) == prefs
        @test m.params[18].set == IntervalSet(0, 1)
        @test m.params[19].set == IntervalSet(0, 1)
    end
    # test for errors
    @testset "Errors" begin
        @test_macro_throws ErrorException @infinite_parameter(m, 0 <= [1:2] <= 1)
        @test_macro_throws ErrorException @infinite_parameter(m, 0 <= "bob" <= 1)
        @test_macro_throws ErrorException @infinite_parameter(m, 0 <= a <= 1)
        @test_macro_throws ErrorException @infinite_parameter(m, 0 <= j)
        @test_macro_throws ErrorException @infinite_parameter(m, j)
        @test_macro_throws ErrorException @infinite_parameter(m, j, foo = 42)
        @test_macro_throws ErrorException @infinite_parameter(m, j in Multinomial(3, 2))
        @test_macro_throws ErrorException @infinite_parameter(m, 0 <= k <= 1, Int)
    end
end

# Test if used
@testset "Used" begin
    m = InfiniteModel()
    param = InfOptParameter(IntervalSet(0, 1), Number[], false)
    pref = add_parameter(m, param, "test")
    # used_by_constraint
    @testset "used_by_constraint" begin
        @test !used_by_constraint(pref)
        m.param_to_constrs[JuMP.index(pref)] = [1]
        @test used_by_constraint(pref)
        delete!(m.param_to_constrs, JuMP.index(pref))
    end
    # used_by_measure
    @testset "used_by_measure" begin
        @test !used_by_measure(pref)
        m.param_to_meas[JuMP.index(pref)] = [1]
        @test used_by_measure(pref)
        delete!(m.param_to_meas, JuMP.index(pref))
    end
    # used_by_variable
    @testset "used_by_variable" begin
        @test !used_by_variable(pref)
        m.param_to_vars[JuMP.index(pref)] = [1]
        @test used_by_variable(pref)
        delete!(m.param_to_vars, JuMP.index(pref))
    end
    # is_used
    @testset "is_used" begin
        @test !is_used(pref)
        m.param_to_constrs[JuMP.index(pref)] = [1]
        @test is_used(pref)
        delete!(m.param_to_constrs, JuMP.index(pref))
        m.param_to_meas[JuMP.index(pref)] = [1]
        @test is_used(pref)
        delete!(m.param_to_meas, JuMP.index(pref))
        m.param_to_vars[JuMP.index(pref)] = [1]
        @test is_used(pref)
        delete!(m.param_to_vars, JuMP.index(pref))
    end
end


# Test parameter set methods
@testset "Infinite Set" begin
    m = InfiniteModel()
    param = InfOptParameter(IntervalSet(0, 1), Number[], false)
    pref = add_parameter(m, param, "test")
    m.param_to_vars[JuMP.index(pref)] = [1]
    # _parameter_set
    @testset "_parameter_set" begin
        @test InfiniteOpt._parameter_set(pref) == IntervalSet(0, 1)
    end
    # _update_parameter_set
    @testset "_update_parameter_set " begin
        @test isa(InfiniteOpt._update_parameter_set(pref,
                                                    IntervalSet(1, 2)), Nothing)
        @test InfiniteOpt._parameter_set(pref) == IntervalSet(1, 2)
    end
    # infinite_set
    @testset "infinite_set" begin
        @test infinite_set(pref) == IntervalSet(1, 2)
    end
    # set_infinite_set
    @testset "set_infinite_set" begin
        @test isa(set_infinite_set(pref, IntervalSet(1, 3)), Nothing)
        @test infinite_set(pref) == IntervalSet(1, 3)
    end
end

# Test parameter support methods
@testset "Supports" begin
    m = InfiniteModel()
    param = InfOptParameter(IntervalSet(0, 1), Number[], false)
    pref = add_parameter(m, param, "test")
    # _parameter_supports
    @testset "_parameter_supports" begin
        @test InfiniteOpt._parameter_supports(pref) == Number[]
    end
    # _update_parameter_supports
    @testset "_update_parameter_supports " begin
        @test isa(InfiniteOpt._update_parameter_supports(pref, [1]), Nothing)
        @test InfiniteOpt._parameter_supports(pref) == [1]
    end
    # num_supports
    @testset "num_supports" begin
        @test num_supports(pref) == 1
    end
    # has_supports
    @testset "has_supports" begin
        @test has_supports(pref)
        InfiniteOpt._update_parameter_supports(pref, Number[])
        @test !has_supports(pref)
    end
    # supports
    @testset "supports" begin
        @test_throws ErrorException supports(pref)
        InfiniteOpt._update_parameter_supports(pref, [1])
        @test supports(pref) == [1]
    end
    # set_supports
    @testset "set_supports" begin
        @test isa(set_supports(pref, [0, 1]), Nothing)
        @test supports(pref) == [0, 1]
        @test_throws ErrorException set_supports(pref, [2, 3])
        warn = "Support points are not unique, eliminating redundant points."
        @test_logs (:warn, warn) set_supports(pref, [1, 1])
    end
    # add_supports
    @testset "add_supports" begin
        @test isa(add_supports(pref, 0.5), Nothing)
        @test supports(pref) == [0.5, 1]
        @test isa(add_supports(pref, [0, 0.25, 1]), Nothing)
        @test supports(pref) == [0, 0.25, 0.5, 1]
    end
    # delete_supports
    @testset "delete_supports" begin
        @test isa(delete_supports(pref), Nothing)
        @test_throws ErrorException supports(pref)
    end
end

# Test lower bound functions
@testset "Lower Bound" begin
    m = InfiniteModel()
    param = InfOptParameter(IntervalSet(0, 1), Number[], false)
    pref = add_parameter(m, param, "test")
    # JuMP.has_lower_bound
    @testset "JuMP.has_lower_bound" begin
        @test has_lower_bound(pref)
        set_infinite_set(pref, DistributionSet(Normal()))
        @test has_lower_bound(pref)
        set_infinite_set(pref, DistributionSet(Multinomial(3, 2)))
        @test !has_lower_bound(pref)
        set_infinite_set(pref, BadSet())
        @test_throws ErrorException has_lower_bound(pref)
    end
    # JuMP.lower_bound
    @testset "JuMP.lower_bound" begin
        set_infinite_set(pref, DistributionSet(Multinomial(3, 2)))
        @test_throws ErrorException lower_bound(pref)
        set_infinite_set(pref, IntervalSet(0, 1))
        @test lower_bound(pref) == 0
        set_infinite_set(pref, DistributionSet(Normal()))
        @test lower_bound(pref) == -Inf
    end
    # JuMP.set_lower_bound
    @testset "JuMP.set_lower_bound" begin
        @test_throws ErrorException set_lower_bound(pref, 2)
        set_infinite_set(pref, BadSet())
        @test_throws ErrorException set_lower_bound(pref, 2)
        set_infinite_set(pref, IntervalSet(0, 1))
        @test isa(set_lower_bound(pref, -1), Nothing)
        @test lower_bound(pref) == -1
    end
end

# Test upper bound functions
@testset "Upper Bound" begin
    m = InfiniteModel()
    param = InfOptParameter(IntervalSet(0, 1), Number[], false)
    pref = add_parameter(m, param, "test")
    # JuMP.has_upper_bound
    @testset "JuMP.has_upper_bound" begin
        @test has_upper_bound(pref)
        set_infinite_set(pref, DistributionSet(Normal()))
        @test has_upper_bound(pref)
        set_infinite_set(pref, DistributionSet(Multinomial(3, 2)))
        @test !has_upper_bound(pref)
        set_infinite_set(pref, BadSet())
        @test_throws ErrorException has_upper_bound(pref)
    end
    # JuMP.upper_bound
    @testset "JuMP.upper_bound" begin
        set_infinite_set(pref, DistributionSet(Multinomial(3, 2)))
        @test_throws ErrorException upper_bound(pref)
        set_infinite_set(pref, IntervalSet(0, 1))
        @test upper_bound(pref) == 1
        set_infinite_set(pref, DistributionSet(Normal()))
        @test upper_bound(pref) == Inf
    end
    # JuMP.set_upper_bound
    @testset "JuMP.set_upper_bound" begin
        @test_throws ErrorException set_upper_bound(pref, 2)
        set_infinite_set(pref, BadSet())
        @test_throws ErrorException set_upper_bound(pref, 2)
        set_infinite_set(pref, IntervalSet(0, 1))
        @test isa(set_upper_bound(pref, 2), Nothing)
        @test upper_bound(pref) == 2
    end
end

# Test the independent manipulation functions
@testset "Independent" begin
    m = InfiniteModel()
    param = InfOptParameter(IntervalSet(0, 1), Number[], false)
    pref = add_parameter(m, param, "test")
    # is_independent
    @testset "is_indepentent" begin
        @test !is_independent(pref)
    end
    # set_independent
    @testset "set_indepentent" begin
        @test isa(set_independent(pref), Nothing)
        @test is_independent(pref)
    end
    # unset_independent
    @testset "unset_indepentent" begin
        @test isa(unset_independent(pref), Nothing)
        @test !is_independent(pref)
    end
end

# Test the internal helper functions
@testset "Internal Helpers" begin
    m = InfiniteModel()
    param = InfOptParameter(IntervalSet(0, 1), Number[], false)
    pref = add_parameter(m, param, "test")
    pref2 = add_parameter(m, param, "θ")
    prefs = @infinite_parameter(m, x[1:2], set = IntervalSet(0, 1),
                                container = SparseAxisArray)
    # group_id
    @testset "group_id" begin
        @test group_id(pref) == 1
    end
    # _group (Parameter)
    @testset "_group (Parameter)" begin
        @test InfiniteOpt._group(pref) == 1
    end
    # _group (Array)
    @testset "_group (Array)" begin
        @test InfiniteOpt._group(prefs) == 3
    end
    # group_id (array)
    @testset "group_id (array)" begin
        @test group_id(prefs) == 3
        @test_throws ErrorException group_id([pref; pref2])
    end
    # _group (over Tuple)
    @testset "_group (Tuple)" begin
        @test InfiniteOpt._group.((prefs, pref)) == (3, 1)
    end
    # _only_one_group
    @testset "_only_one_group" begin
        @test InfiniteOpt._only_one_group(pref)
        @test InfiniteOpt._only_one_group(prefs)
        @test !InfiniteOpt._only_one_group(convert(JuMP.Containers.SparseAxisArray,
                                                                 [pref2; pref]))
    end
    # _root_name
    @testset "_root_name" begin
        @test InfiniteOpt._root_name(pref) == "test"
        @test InfiniteOpt._root_name(prefs[1]) == "x"
        @test InfiniteOpt._root_name(pref2) == "θ"
    end
    # _root_names
    @testset "_root_names" begin
        @test InfiniteOpt._root_names((pref, prefs, pref2)) == ("test", "x", "θ")
    end
    # _list_parameter_refs
    @testset "_list_parameter_refs" begin
        @test InfiniteOpt._list_parameter_refs((pref, prefs)) isa Vector
        result = InfiniteOpt._list_parameter_refs((pref, prefs))
        @test length(result) == 3
        @test result[1] == pref
        @test result[2] == prefs[1] || result[2] == prefs[2]
        @test result[3] == prefs[1] || result[3] == prefs[2]
    end
end

# Test everything else
@testset "Other Queries" begin
    m = InfiniteModel()
    param = InfOptParameter(IntervalSet(0, 1), Number[], false)
    pref = add_parameter(m, param, "test")
    prefs = @infinite_parameter(m, [1:2], set = IntervalSet(0, 1),
                                container = SparseAxisArray)
    # JuMP.is_valid
    @testset "JuMP.is_valid" begin
        @test is_valid(m, pref)
        pref2 = InfOptVariableRef(InfiniteModel(), 1, Parameter)
        @test !is_valid(m, pref2)
        pref3 = InfOptVariableRef(m, 5, Parameter)
        @test !is_valid(m, pref3)
    end
    # supports (array)
    @testset "supports (array)" begin
        @test_throws ErrorException supports(prefs)
        @test_throws ErrorException supports([prefs[1]; pref])
        set_supports(prefs[1], [0])
        set_supports(prefs[2], [0, 1])
        @test_throws ErrorException supports(prefs)
        add_supports(prefs[1], [1])
        expected = JuMP.Containers.SparseAxisArray[]
        push!(expected, convert(JuMP.Containers.SparseAxisArray, [0; 0]))
        push!(expected, convert(JuMP.Containers.SparseAxisArray, [1; 1]))
        @test supports(prefs) == expected
        expected = JuMP.Containers.SparseAxisArray[]
        push!(expected, convert(JuMP.Containers.SparseAxisArray, [0; 0]))
        push!(expected, convert(JuMP.Containers.SparseAxisArray, [0; 1]))
        push!(expected, convert(JuMP.Containers.SparseAxisArray, [1; 0]))
        push!(expected, convert(JuMP.Containers.SparseAxisArray, [1; 1]))
        set_independent(prefs[1])
        set_independent(prefs[2])
        @test sort(supports(prefs)) == sort(expected)
    end
    # num_parameters
    @testset "num_parameters" begin
        @test num_parameters(m) == 3
        delete!(m.params, JuMP.index(pref))
        @test num_parameters(m) == 2
        m.params[JuMP.index(pref)] = param
    end
    # all_parameters
    @testset "all_parameters" begin
        @test length(all_parameters(m)) == 3
        @test all_parameters(m)[1] == pref
    end
end
