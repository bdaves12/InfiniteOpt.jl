# Test helper functions for infinite variable macro
@testset "Infinite Helpers" begin
    # _check_rhs
    @testset "_check_rhs" begin
        # test with reversed sides
        arg1 = :(data[i]); arg2 = :(x[i=1:2])
        @test InfiniteOpt._check_rhs(arg1, arg2) == (arg2, arg1)
        # test with normal case that shouldn't be swapped
        arg1 = :((x[i=1:2])(t)); arg2 = :(data[i])
        @test InfiniteOpt._check_rhs(arg1, arg2) == (arg1, arg2)
        # test reversed case that stays reversed because cannot be distinguished
        arg1 = :(data(i)); arg2 = :((x[i=1:2])(t))
        @test InfiniteOpt._check_rhs(arg1, arg2) == (arg1, arg2)
    end
    # _less_than_parse
    @testset "_less_than_parse" begin
        # test with reversed sides
        arg1 = :(data[i]); arg2 = :(x[i=1:2])
        @test InfiniteOpt._less_than_parse(arg1, arg2) == (:($(arg2) >= $(arg1)),
                                                           nothing)
        # test normal reference with parameter tuple
        arg1 = :(x[i=1:2](t)); arg2 = :(data[i])
        @test InfiniteOpt._less_than_parse(arg1, arg2) == (:(x[i=1:2] <= $(arg2)),
                                                           :((t,)))
        # test normal with parameter tuple
        arg1 = :(x(t)); arg2 = :(data[i])
        @test InfiniteOpt._less_than_parse(arg1, arg2) == (:(x <= $(arg2)),
                                                           :((t,)))
        # test normal without tuple
        arg1 = :(x); arg2 = :(data[i])
        @test InfiniteOpt._less_than_parse(arg1, arg2) == (:(x <= $(arg2)),
                                                           nothing)
        # test normal without tuple
        arg1 = :(x); arg2 = 1
        @test InfiniteOpt._less_than_parse(arg1, arg2) == (:(x <= $(arg2)),
                                                           nothing)
    end
    # _greater_than_parse
    @testset "_greater_than_parse" begin
        # test with reversed sides
        arg1 = :(data[i]); arg2 = :(x[i=1:2])
        @test InfiniteOpt._greater_than_parse(arg1, arg2) == (:($(arg2) <= $(arg1)),
                                                              nothing)
        # test normal reference with parameter tuple
        arg1 = :(x[i=1:2](t)); arg2 = :(data[i])
        @test InfiniteOpt._greater_than_parse(arg1, arg2) == (:(x[i=1:2] >= $(arg2)),
                                                              :((t,)))
        # test normal with parameter tuple
        arg1 = :(x(t)); arg2 = :(data[i])
        @test InfiniteOpt._greater_than_parse(arg1, arg2) == (:(x >= $(arg2)),
                                                              :((t,)))
        # test normal without tuple
        arg1 = :(x); arg2 = :(data[i])
        @test InfiniteOpt._greater_than_parse(arg1, arg2) == (:(x >= $(arg2)),
                                                              nothing)
        # test normal without tuple
        arg1 = :(x); arg2 = 1
        @test InfiniteOpt._greater_than_parse(arg1, arg2) == (:(x >= $(arg2)),
                                                              nothing)
    end
    # _less_than_parse (number on lhs)
    @testset "_less_than_parse (reversed)" begin
        # test with reference
        arg1 = 1; arg2 = :(x[i=1:2])
        @test InfiniteOpt._less_than_parse(arg1, arg2) == (:($(arg2) >= $(arg1)),
                                                           nothing)
        # test with reference and parameter tuple
        arg1 = 1; arg2 = :(x[i=1:2](t))
        @test InfiniteOpt._less_than_parse(arg1, arg2) == (:(x[i=1:2] >= $(arg1)),
                                                           :((t,)))
        # test normal without tuple
        arg1 = 1; arg2 = :(x)
        @test InfiniteOpt._less_than_parse(arg1, arg2) == (:(x >= $(arg1)),
                                                           nothing)
    end
    # _greater_than_parse (number on lhs)
    @testset "_greater_than_parse (reversed)" begin
        # test with reference
        arg1 = 1; arg2 = :(x[i=1:2])
        @test InfiniteOpt._greater_than_parse(arg1, arg2) == (:($(arg2) <= $(arg1)),
                                                              nothing)
        # test with reference and parameter tuple
        arg1 = 1; arg2 = :(x[i=1:2](t))
        @test InfiniteOpt._greater_than_parse(arg1, arg2) == (:(x[i=1:2] <= $(arg1)),
                                                              :((t,)))
        # test normal without tuple
        arg1 = 1; arg2 = :(x)
        @test InfiniteOpt._greater_than_parse(arg1, arg2) == (:(x <= $(arg1)),
                                                              nothing)
    end
    # _equal_to_parse
    @testset "_equal_to_parse" begin
        # test with reversed sides
        arg1 = :(data[i]); arg2 = :(x[i=1:2])
        @test InfiniteOpt._equal_to_parse(arg1, arg2) == (:($(arg2) == $(arg1)),
                                                          nothing)
        # test normal reference with parameter tuple
        arg1 = :(x[i=1:2](t)); arg2 = :(data[i])
        @test InfiniteOpt._equal_to_parse(arg1, arg2) == (:(x[i=1:2] == $(arg2)),
                                                          :((t,)))
        # test normal with parameter tuple
        arg1 = :(x(t)); arg2 = :(data[i])
        @test InfiniteOpt._equal_to_parse(arg1, arg2) == (:(x == $(arg2)),
                                                          :((t,)))
        # test normal without tuple
        arg1 = :(x); arg2 = :(data[i])
        @test InfiniteOpt._equal_to_parse(arg1, arg2) == (:(x == $(arg2)),
                                                          nothing)
        # test normal without tuple
        arg1 = :(x); arg2 = 1
        @test InfiniteOpt._equal_to_parse(arg1, arg2) == (:(x == $(arg2)),
                                                          nothing)
    end
    # _equal_to_parse (number on lhs)
    @testset "_equal_to_parse (reversed)" begin
        # test with reference
        arg1 = 1; arg2 = :(x[i=1:2])
        @test InfiniteOpt._equal_to_parse(arg1, arg2) == (:($(arg2) == $(arg1)),
                                                          nothing)
        # test with reference and parameter tuple
        arg1 = 1; arg2 = :(x[i=1:2](t))
        @test InfiniteOpt._equal_to_parse(arg1, arg2) == (:(x[i=1:2] == $(arg1)),
                                                          :((t,)))
        # test normal without tuple
        arg1 = 1; arg2 = :(x)
        @test InfiniteOpt._equal_to_parse(arg1, arg2) == (:(x == $(arg1)),
                                                          nothing)
    end
    # _parse_parameters (call)
    @testset "_parse_parameters (call)" begin
        # test less than parse
        expr = :(x(t, x) <= 1)
        @test InfiniteOpt._parse_parameters(error, Val(expr.head),
                                            expr.args) == (:(x <= 1), :((t, x)))
        # test greater than parse
        expr = :(x[1:2](t) >= 1)
        @test InfiniteOpt._parse_parameters(error, Val(expr.head),
                                         expr.args) == (:(x[1:2] >= 1), :((t,)))
        # test equal to parse
        expr = :(x(t) == d)
        @test InfiniteOpt._parse_parameters(error, Val(expr.head),
                                            expr.args) == (:(x == d), :((t,)))
        # test only variable parse
        expr = :(x(t))
        @test InfiniteOpt._parse_parameters(error, Val(expr.head),
                                            expr.args) == (:(x), :((t,)))
        # test invalid use of operator
        expr = :(x(t) in 1)
        @test_throws ErrorException InfiniteOpt._parse_parameters(error,
                                                                  Val(expr.head),
                                                                  expr.args)
    end
    # _parse_parameters (comparison)
    @testset "_parse_parameters (compare)" begin
        # test with parameters
        expr = :(0 <= x(t, x) <= 1)
        @test InfiniteOpt._parse_parameters(error, Val(expr.head),
                                       expr.args) == (:(0 <= x <= 1), :((t, x)))
        # test with parameters and references
        expr = :(0 <= x[1:2](t, x) <= 1)
        @test InfiniteOpt._parse_parameters(error, Val(expr.head),
                                  expr.args) == (:(0 <= x[1:2] <= 1), :((t, x)))
        # test without parameters
        expr = :(0 <= x <= 1)
        @test InfiniteOpt._parse_parameters(error, Val(expr.head),
                                         expr.args) == (:(0 <= x <= 1), nothing)
    end
end

# Test the infinite variable macro
@testset "Infinite" begin
    # initialize model and infinite parameters
    m = InfiniteModel()
    @infinite_parameter(m, 0 <= t <= 1)
    @infinite_parameter(m, -1 <= x[1:2] <= 1)
    # test single variable definition
    @testset "Single" begin
        # test basic defaults
        vref = InfOptVariableRef(m, 1, Infinite)
        @test @infinite_variable(m, parameter_refs = t) == vref
        @test name(vref) == "noname(t)"
        # test more tuple input and variable details
        vref = InfOptVariableRef(m, 2, Infinite)
        @test @infinite_variable(m, parameter_refs = (t, x), base_name = "test",
                                 binary = true) == vref
        @test name(vref) == "test(t, x)"
        @test is_binary(vref)
        # test nonanonymous with simple single arg
        vref = InfOptVariableRef(m, 3, Infinite)
        @test @infinite_variable(m, a(x)) == vref
        @test name(vref) == "a(x)"
        # test nonanonymous with complex single arg
        vref = InfOptVariableRef(m, 4, Infinite)
        @test @infinite_variable(m, 0 <= b(x) <= 1) == vref
        @test name(vref) == "b(x)"
        @test lower_bound(vref) == 0
        @test upper_bound(vref) == 1
        # test nonanonymous with reversed single arg
        vref = InfOptVariableRef(m, 5, Infinite)
        @test @infinite_variable(m, 0 <= c(t)) == vref
        @test name(vref) == "c(t)"
        @test lower_bound(vref) == 0
        # test multi-argument expr 1
        vref = InfOptVariableRef(m, 6, Infinite)
        @test @infinite_variable(m, d(t) == 0, Int, base_name = "test") == vref
        @test name(vref) == "test(t)"
        @test fix_value(vref) == 0
    end
    # test array variable definition
    @testset "Array" begin
        # test anonymous array
        vrefs = [InfOptVariableRef(m, 7, Infinite), InfOptVariableRef(m, 8, Infinite)]
        @test @infinite_variable(m, [1:2], parameter_refs = t) == vrefs
        @test name(vrefs[1]) == "noname(t)"
        # test basic param expression
        vrefs = [InfOptVariableRef(m, 9, Infinite), InfOptVariableRef(m, 10, Infinite)]
        @test @infinite_variable(m, e[1:2], parameter_refs = (t, x)) == vrefs
        @test name(vrefs[2]) == "e[2](t, x)"
        # test comparison without params
        vrefs = [InfOptVariableRef(m, 11, Infinite), InfOptVariableRef(m, 12, Infinite)]
        @test @infinite_variable(m, 0 <= f[1:2] <= 1,
                                 parameter_refs = (t, x)) == vrefs
        @test name(vrefs[2]) == "f[2](t, x)"
        @test lower_bound(vrefs[1]) == 0
        @test upper_bound(vrefs[2]) == 1
        # test comparison with call
        vrefs = [InfOptVariableRef(m, 13, Infinite), InfOptVariableRef(m, 14, Infinite)]
        @test @infinite_variable(m, 0 <= g[1:2](t) <= 1) == vrefs
        @test name(vrefs[1]) == "g[1](t)"
        @test lower_bound(vrefs[1]) == 0
        @test upper_bound(vrefs[2]) == 1
        # test fixed
        vrefs = [InfOptVariableRef(m, 15, Infinite), InfOptVariableRef(m, 16, Infinite)]
        @test @infinite_variable(m, h[i = 1:2](t) == ones(2)[i]) == vrefs
        @test name(vrefs[1]) == "h[1](t)"
        @test fix_value(vrefs[1]) == 1
        # test containers
        vrefs = [InfOptVariableRef(m, 17, Infinite), InfOptVariableRef(m, 18, Infinite)]
        vrefs = convert(JuMP.Containers.SparseAxisArray, vrefs)
        @test @infinite_variable(m, [1:2](t),
                                 container = SparseAxisArray) == vrefs
        @test name(vrefs[1]) == "noname(t)"
    end
    # test errors
    @testset "Errors" begin
        # test model assertion errors
        m2 = Model()
        @test_throws AssertionError @infinite_variable(m2, parameter_refs = t)
        @test_throws AssertionError @infinite_variable(m2, i(t))
        @test_throws AssertionError @infinite_variable(m2, i, Int)
        @test_throws AssertionError @infinite_variable(m2, i(t), Bin)
        # test double specification
        @test_macro_throws ErrorException @infinite_variable(m, i(t),
                                                             parameter_refs = x)
        # test undefined parameter error
        @test_macro_throws ErrorException @infinite_variable(m, i, Int)
        # test invalid keyword arguments
        @test_macro_throws ErrorException @infinite_variable(m, i(t),
                                                           parameter_values = 1)
        @test_macro_throws ErrorException @infinite_variable(m, i(t),
                      infinite_variable_ref = InfOptVariableRef(m, 1, Infinite))
        @test_macro_throws ErrorException @infinite_variable(m, i(t), bad = 42)
        # test name duplication
        @test_macro_throws ErrorException @infinite_variable(m, a(t), Int)
    end
end

# Test the point variable macro
@testset "Point" begin
    # initialize model, parameters, and infinite variables
    m = InfiniteModel()
    @infinite_parameter(m, 0 <= t <= 1)
    @infinite_parameter(m, -1 <= x[1:2] <= 1)
    @infinite_variable(m, 0 <= z(t, x) <= 1, Int)
    @infinite_variable(m, z2[1:2](t) == 3)
    # test single variable definition
    @testset "Single" begin
        # test simple anon case
        vref = InfOptVariableRef(m, 4, Point)
        @test @point_variable(m, infinite_variable_ref = z,
                              parameter_values = (0, [0, 0])) == vref
        @test infinite_variable_ref(vref) == z
        pt = convert(JuMP.Containers.SparseAxisArray, zeros(2))
        @test parameter_values(vref) == (0, pt)
        @test is_integer(vref)
        @test lower_bound(vref) == 0
        # test anon with changes to fixed
        vref = InfOptVariableRef(m, 5, Point)
        @test @point_variable(m, infinite_variable_ref = z, lower_bound = -5,
                          parameter_values = (0, [0, 0]), binary = true) == vref
        @test infinite_variable_ref(vref) == z
        @test parameter_values(vref) == (0, pt)
        @test !is_integer(vref)
        @test is_binary(vref)
        @test lower_bound(vref) == -5
        # test regular with alias
        vref = InfOptVariableRef(m, 6, Point)
        @test @point_variable(m, z(0, [0, 0]), z0, Bin) == vref
        @test infinite_variable_ref(vref) == z
        @test parameter_values(vref) == (0, pt)
        @test is_binary(vref)
        @test lower_bound(vref) == 0
        @test name(vref) == "z0"
        # test regular with semi anon
        vref = InfOptVariableRef(m, 7, Point)
        @test @point_variable(m, z(0, [0, 0]), base_name = "z0",
                              binary = true) == vref
        @test infinite_variable_ref(vref) == z
        @test parameter_values(vref) == (0, pt)
        @test is_binary(vref)
        @test lower_bound(vref) == 0
        @test name(vref) == "z0"
    end
    # test array variable definition
    @testset "Array" begin
        # test anon array with one infvar
        vrefs = [InfOptVariableRef(m, 8, Point), InfOptVariableRef(m, 9, Point)]
        @test @point_variable(m, [1:2], infinite_variable_ref = z,
                              parameter_values = (0, [0, 0])) == vrefs
        @test infinite_variable_ref(vrefs[1]) == z
        pt = convert(JuMP.Containers.SparseAxisArray, zeros(2))
        @test parameter_values(vrefs[2]) == (0, pt)
        @test is_integer(vrefs[1])
        @test lower_bound(vrefs[2]) == 0
        # test anon array with different inf vars
        vrefs = [InfOptVariableRef(m, 10, Point), InfOptVariableRef(m, 11, Point)]
        @test @point_variable(m, [i = 1:2], infinite_variable_ref = z2[i],
                              parameter_values = 0) == vrefs
        @test infinite_variable_ref(vrefs[1]) == z2[1]
        @test infinite_variable_ref(vrefs[2]) == z2[2]
        @test parameter_values(vrefs[2]) == (0,)
        @test fix_value(vrefs[2]) == 3
        @test name(vrefs[1]) == "z2[1](0)"
        # test array with same infvar
        vrefs = [InfOptVariableRef(m, 12, Point), InfOptVariableRef(m, 13, Point)]
        @test @point_variable(m, z(0, [0, 0]), a[1:2], Bin) == vrefs
        @test infinite_variable_ref(vrefs[1]) == z
        @test parameter_values(vrefs[2]) == (0, pt)
        @test is_binary(vrefs[1])
        @test lower_bound(vrefs[2]) == 0
        @test name(vrefs[1]) == "a[1]"
        # test test array with differnt infvars
        vrefs = [InfOptVariableRef(m, 14, Point), InfOptVariableRef(m, 15, Point)]
        @test @point_variable(m, z2[i](0), b[i = 1:2] >= -5) == vrefs
        @test infinite_variable_ref(vrefs[1]) == z2[1]
        @test infinite_variable_ref(vrefs[2]) == z2[2]
        @test parameter_values(vrefs[2]) == (0,)
        @test lower_bound(vrefs[2]) == -5
        @test name(vrefs[1]) == "b[1]"
        # test semi anon array
        vrefs = [InfOptVariableRef(m, 16, Point), InfOptVariableRef(m, 17, Point)]
        @test @point_variable(m, z2[i](0), [i = 1:2], lower_bound = -5) == vrefs
        @test infinite_variable_ref(vrefs[1]) == z2[1]
        @test infinite_variable_ref(vrefs[2]) == z2[2]
        @test lower_bound(vrefs[2]) == -5
        @test name(vrefs[1]) == "z2[1](0)"
    end
    # test errors
    @testset "Errors" begin
        # test model assertion errors
        m2 = Model()
        @test_throws AssertionError @point_variable(m2, infinite_variable_ref = z,
                                                 parameter_values = (0, [0, 0]))
        @test_throws AssertionError @point_variable(m2, z(0, [0, 0]), bob, Bin)
        @test_throws AssertionError @point_variable(m2, [1:2],
                      infinite_variable_ref = z, parameter_values = (0, [0, 0]))
        # test double specification
        @test_macro_throws ErrorException @point_variable(m, z(0, [0, 0]), bob,
                                                      infinite_variable_ref = z)
        @test_macro_throws ErrorException @point_variable(m, z(0, [0, 0]), bob,
                                                 parameter_values = (0, [0, 0]))
        # test the adding expressions to infvar
        @test_macro_throws ErrorException @point_variable(m, z(0, [0, 0]) >= 0,
                                                          bob, Bin)
        @test_macro_throws ErrorException @point_variable(m, z(0, [0, 0]) == 0,
                                                          bob, Bin)
        @test_macro_throws ErrorException @point_variable(m, z(0, [0, 0]) in 0,
                                                          bob, Bin)
        # test other syntaxes
        @test_macro_throws ErrorException @point_variable(m,
                                               0 <= z(0, [0, 0]) <= 1, bob, Bin)
        @test_macro_throws ErrorException @point_variable(m, [1:2], Bin,
                      infinite_variable_ref = z, parameter_values = (0, [0, 0]))
        @test_macro_throws ErrorException @point_variable(m, bob,
                     infinite_variable_ref = z, parameter_values = (0, [0, 0]))
        # test redefinition catch
        @test_macro_throws ErrorException @point_variable(m, z(0, [0, 0]), z0,
                                                          Bin)
    end
end

# Test the global variable macro
@testset "Global" begin
    # initialize model
    m = InfiniteModel()
    # test regular
    vref = InfOptVariableRef(m, 1, Global)
    @test @global_variable(m, x >= 1, Bin) == vref
    @test name(vref) == "x"
    @test lower_bound(vref) == 1
    @test is_binary(vref)
    # test anan
    vref = InfOptVariableRef(m, 2, Global)
    @test @global_variable(m, binary = true, lower_bound = 1,
                           base_name = "x") == vref
    @test name(vref) == "x"
    @test lower_bound(vref) == 1
    @test is_binary(vref)
    # test array
    vrefs = [InfOptVariableRef(m, 3, Global), InfOptVariableRef(m, 4, Global)]
    @test @global_variable(m, y[1:2] == 2, Int) == vrefs
    @test name(vrefs[1]) == "y[1]"
    @test fix_value(vrefs[2]) == 2
    @test is_integer(vrefs[1])
    # test errors
    @test_throws AssertionError @global_variable(Model(), z >= 1, Bin)
    @test_macro_throws ErrorException @global_variable(m, x >= 1, Bin)
end
