# Test name methods
@testset "Basics" begin
    # setup data
    m = InfiniteModel()
    @independent_parameter(m, a in [0, 1])
    @independent_parameter(m, b[1:3] in [0, 1])
    @dependent_parameters(m, c[1:2] in [0, 1])
    @independent_parameter(m, d in [0, 1])
    idx = InfiniteVariableIndex(1)
    num = Float64(0)
    info = VariableInfo(false, num, false, num, false, num, false, num, false, false)
    new_info = VariableInfo(true, 0., true, 0., true, 0., true, 0., true, false)
    var = InfiniteVariable(info, VectorTuple(a, b[1:2], c, [b[3], d]),
                           [1:7...], [1:6...])
    object = VariableData(var, "var")
    vref = InfiniteVariableRef(m, idx)
    gvref = GeneralVariableRef(m, 1, InfiniteVariableIndex)
    # JuMP.owner_model
    @testset "JuMP.owner_model" begin
        @test owner_model(vref) === m
        @test owner_model(gvref) === m
    end
    # JuMP.index
    @testset "JuMP.index" begin
        @test index(vref) == idx
        @test index(gvref) == idx
    end
    # dispatch_variable_ref
    @testset "dispatch_variable_ref" begin
        @test dispatch_variable_ref(m, idx) == vref
        @test dispatch_variable_ref(gvref) == vref
    end
    # _add_data_object
    @testset "_add_data_object" begin
        @test InfiniteOpt._add_data_object(m, object) == idx
    end
    # _data_dictionary
    @testset "_data_dictionary" begin
        @test InfiniteOpt._data_dictionary(m, InfiniteVariable) === m.infinite_vars
        @test InfiniteOpt._data_dictionary(vref) === m.infinite_vars
        @test InfiniteOpt._data_dictionary(gvref) === m.infinite_vars
    end
    # JuMP.is_valid
    @testset "JuMP.is_valid" begin
        @test is_valid(m, vref)
        @test is_valid(m, gvref)
    end
    # _data_object
    @testset "_data_object" begin
        @test InfiniteOpt._data_object(vref) === object
        @test InfiniteOpt._data_object(gvref) === object
    end
    # _core_variable_object
    @testset "_core_variable_object" begin
        @test InfiniteOpt._core_variable_object(vref) === var
        @test InfiniteOpt._core_variable_object(gvref) === var
    end
    # _set_core_variable_object
    @testset "_set_core_variable_object" begin
        @test InfiniteOpt._set_core_variable_object(vref, var) isa Nothing
    end
    # _object_numbers
    @testset "_object_numbers" begin
        @test InfiniteOpt._object_numbers(vref) == [1:6...]
    end
    # _parameter_numbers
    @testset "_parameter_numbers" begin
        @test InfiniteOpt._parameter_numbers(vref) == [1:7...]
    end
    @testset "_variable_info" begin
        @test InfiniteOpt._variable_info(vref) == info
    end
    # _update_variable_info
    @testset "_update_variable_info" begin
        @test isa(InfiniteOpt._update_variable_info(vref, new_info), Nothing)
        @test InfiniteOpt._variable_info(vref) == new_info
    end
    # _measure_dependencies
    @testset "_measure_dependencies" begin
        @test InfiniteOpt._measure_dependencies(vref) == MeasureIndex[]
        @test InfiniteOpt._measure_dependencies(gvref) == MeasureIndex[]
    end
    # _constraint_dependencies
    @testset "_constraint_dependencies" begin
        @test InfiniteOpt._constraint_dependencies(vref) == ConstraintIndex[]
        @test InfiniteOpt._constraint_dependencies(gvref) == ConstraintIndex[]
    end
    # _reduced_variable_dependencies
    @testset "_reduced_variable_dependencies" begin
        @test InfiniteOpt._reduced_variable_dependencies(vref) == ReducedInfiniteVariableIndex[]
        @test InfiniteOpt._reduced_variable_dependencies(gvref) == ReducedInfiniteVariableIndex[]
    end
    # _point_variable_dependencies
    @testset "_point_variable_dependencies" begin
        @test InfiniteOpt._point_variable_dependencies(vref) == PointVariableIndex[]
        @test InfiniteOpt._point_variable_dependencies(gvref) == PointVariableIndex[]
    end
    # JuMP.name
    @testset "JuMP.name" begin
        @test name(vref) == "var"
        @test name(gvref) == "var"
    end
    # raw_parameter_refs
    @testset "raw_parameter_refs" begin
        @test raw_parameter_refs(vref) == VectorTuple(a, b[1:2], c, [b[3], d])
        @test raw_parameter_refs(gvref) == VectorTuple(a, b[1:2], c, [b[3], d])
    end
    # parameter_refs
    @testset "parameter_refs" begin
        @test parameter_refs(vref) == (a, b[1:2], c, [b[3], d])
        @test parameter_refs(gvref) == (a, b[1:2], c, [b[3], d])
    end
    # parameter_list
    @testset "parameter_list" begin
        @test parameter_list(vref) == [a; b[1:2]; c; [b[3], d]]
        @test parameter_list(gvref) == [a; b[1:2]; c; [b[3], d]]
        @test parameter_list(raw_parameter_refs(vref)) == [a; b[1:2]; c; [b[3], d]]
    end
    # JuMP.set_name
    @testset "JuMP.set_name" begin
        # test default
        @test isa(set_name(vref, ""), Nothing)
        @test name(vref) == "noname(a, b, c, [b[3], d])"
        # test normal
        @test isa(set_name(gvref, "new"), Nothing)
        @test name(vref) == "new(a, b, c, [b[3], d])"
    end
    # _make_variable_ref
    @testset "_make_variable_ref" begin
        @test InfiniteOpt._make_variable_ref(m, idx) == gvref
    end
    # _var_name_dict
    @testset "_var_name_dict" begin
        @test InfiniteOpt._var_name_dict(m) isa Nothing
    end
    # _update_var_name_dict
    @testset "_update_var_name_dict" begin
        m.name_to_var = Dict{String, ObjectIndex}()
        dict = InfiniteOpt._data_dictionary(vref)
        @test InfiniteOpt._update_var_name_dict(m, dict) isa Nothing
        @test InfiniteOpt._var_name_dict(m) == Dict(name(vref) => idx)
        m.name_to_var = nothing
    end
    # parameter_by_name
    @testset "JuMP.variable_by_name" begin
        # test normal
        @test variable_by_name(m, "new(a, b, c, [b[3], d])") == gvref
        @test variable_by_name(m, "test(a, b, c, [b[3], d])") isa Nothing
        # prepare variable with same name
        idx2 = InfiniteVariableIndex(2)
        @test InfiniteOpt._add_data_object(m, object) == idx2
        vref2 = InfiniteVariableRef(m, idx2)
        @test set_name(vref2, "new") isa Nothing
        # test multiple name error
        @test_throws ErrorException variable_by_name(m, "new(a, b, c, [b[3], d])")
    end
    # _root_name
    @testset "_root_name" begin
        @test InfiniteOpt._root_name(vref) == "new"
    end
    # _delete_data_object
    @testset "_delete_data_object" begin
        @test InfiniteOpt._delete_data_object(vref) isa Nothing
        @test length(InfiniteOpt._data_dictionary(vref)) == 1
        @test !is_valid(m, vref)
    end
end

# Test variable definition methods
@testset "Definition" begin
    # initialize model and infinite variable info
    m = InfiniteModel()
    @independent_parameter(m, pref in [0, 1])
    @independent_parameter(m, pref2 in [0, 1])
    @dependent_parameters(m, prefs[1:2] in [0, 1])
    @finite_parameter(m, fin, 42)
    num = Float64(0)
    info = VariableInfo(false, num, false, num, false, num, false, num, false, false)
    info2 = VariableInfo(true, num, true, num, true, num, true, num, true, false)
    info3 = VariableInfo(true, num, true, num, true, num, true, num, false, true)
    # _check_tuple_element (IndependentParameterRefs)
    @testset "_check_tuple_element (IndependentParameterRefs)" begin
        iprefs = dispatch_variable_ref.([pref, pref2])
        @test InfiniteOpt._check_tuple_element(error, iprefs) isa Nothing
    end
    # _check_tuple_element (DependentParameterRefs)
    @testset "_check_tuple_element (DependentParameterRefs)" begin
        dprefs = dispatch_variable_ref.(prefs)
        @test InfiniteOpt._check_tuple_element(error, dprefs) isa Nothing
        @test_throws ErrorException InfiniteOpt._check_tuple_element(error, dprefs[1:1])
    end
    # _check_tuple_element (Fallback)
    @testset "_check_tuple_element (Fallback)" begin
        refs = dispatch_variable_ref.([fin, pref])
        @test_throws ErrorException InfiniteOpt._check_tuple_element(error, refs)
    end
    # _check_parameter_tuple
    @testset "_check_parameter_tuple" begin
        # test normal
        tuple = VectorTuple((pref, prefs, pref2))
        @test InfiniteOpt._check_parameter_tuple(error, tuple) isa Nothing
        tuple = VectorTuple(([pref, pref2], prefs))
        @test InfiniteOpt._check_parameter_tuple(error, tuple) isa Nothing
        # test bad
        tuple = VectorTuple((pref, pref))
        @test_throws ErrorException InfiniteOpt._check_parameter_tuple(error, tuple)
        tuple = VectorTuple((pref, [prefs[1], pref2]))
        @test_throws ErrorException InfiniteOpt._check_parameter_tuple(error, tuple)
        tuple = VectorTuple(fin)
        @test_throws ErrorException InfiniteOpt._check_parameter_tuple(error, tuple)
    end
    # _make_variable
    @testset "_make_variable" begin
        # test for each error message
        @test_throws ErrorException InfiniteOpt._make_variable(error, info, Val(Infinite),
                                                   bob = 42)
        @test_throws ErrorException InfiniteOpt._make_variable(error, info, Val(:bad))
        @test_throws ErrorException InfiniteOpt._make_variable(error, info, Val(Infinite))
        @test_throws ErrorException InfiniteOpt._make_variable(error, info, Val(Infinite),
                                                   parameter_refs = (pref, fin))
        @test_throws ErrorException InfiniteOpt._make_variable(error, info, Val(Infinite),
                                                  parameter_refs = (pref, pref))
        # defined expected output
        expected = InfiniteVariable(info, VectorTuple(pref), [1], [1])
        # test for expected output
        @test InfiniteOpt._make_variable(error, info, Val(Infinite),
                             parameter_refs = pref).info == expected.info
        @test InfiniteOpt._make_variable(error, info, Val(Infinite),
                parameter_refs = pref).parameter_refs == expected.parameter_refs
        @test InfiniteOpt._make_variable(error, info, Val(Infinite),
                parameter_refs = pref).parameter_nums == expected.parameter_nums
        @test InfiniteOpt._make_variable(error, info, Val(Infinite),
                parameter_refs = pref).object_nums == expected.object_nums
        # test various types of param tuples
        tuple = VectorTuple(pref, pref2)
        @test InfiniteOpt._make_variable(error, info, Val(Infinite),
                 parameter_refs = (pref, pref2)).parameter_refs == tuple
        tuple = VectorTuple(pref, prefs)
        @test InfiniteOpt._make_variable(error, info, Val(Infinite),
                         parameter_refs = (pref, prefs)).parameter_refs == tuple
        @test InfiniteOpt._make_variable(error, info, Val(Infinite),
                         parameter_refs = (pref, prefs)).parameter_nums == [1, 3, 4]
        @test sort!(InfiniteOpt._make_variable(error, info, Val(Infinite),
                         parameter_refs = (pref, prefs)).object_nums) == [1, 3]
        tuple = VectorTuple(prefs)
        @test InfiniteOpt._make_variable(error, info, Val(Infinite),
                             parameter_refs = prefs).parameter_refs == tuple
    end
    # build_variable
    @testset "JuMP.build_variable" begin
        # test for each error message
        @test_throws ErrorException build_variable(error, info, Infinite,
                                                   bob = 42)
        @test_throws ErrorException build_variable(error, info, :bad)
        @test_throws ErrorException build_variable(error, info, Point,
                                                   parameter_refs = pref)
        @test_throws ErrorException build_variable(error, info, Infinite)
        @test_throws ErrorException build_variable(error, info, Infinite,
                                                   parameter_refs = (pref, fin))
        @test_throws ErrorException build_variable(error, info, Infinite,
                                                  parameter_refs = (pref, pref),
                                                  error = error)
        # defined expected output
        expected = InfiniteVariable(info, VectorTuple(pref), [1], [1])
        # test for expected output
        @test build_variable(error, info, Infinite,
                             parameter_refs = pref).info == expected.info
        @test build_variable(error, info, Infinite,
                parameter_refs = pref).parameter_refs == expected.parameter_refs
        # test various types of param tuples
        tuple = VectorTuple(pref, prefs)
        @test build_variable(error, info, Infinite,
                         parameter_refs = (pref, prefs)).parameter_refs == tuple
        tuple = VectorTuple(prefs)
        @test build_variable(error, info, Infinite,
                             parameter_refs = prefs).parameter_refs == tuple
        @test build_variable(error, info, Infinite,
                             parameter_refs = prefs).parameter_nums == [3, 4]
        @test build_variable(error, info, Infinite,
                             parameter_refs = prefs).object_nums == [3]
    end
    # _check_parameters_valid
    @testset "_check_parameters_valid" begin
        # prepare param tuple
        @independent_parameter(InfiniteModel(), pref3 in [0, 1])
        tuple = VectorTuple((pref, prefs, pref3))
        # test that catches error
        @test_throws VariableNotOwned{GeneralVariableRef} InfiniteOpt._check_parameters_valid(m, tuple)
        # test normal
        tuple = VectorTuple(pref, pref2)
        @test InfiniteOpt._check_parameters_valid(m, tuple) isa Nothing
    end
    # _update_param_var_mapping
    @testset "_update_param_var_mapping" begin
        # prepare tuple
        tuple = VectorTuple(pref, prefs)
        idx = InfiniteVariableIndex(1)
        ivref = InfiniteVariableRef(m, idx)
        # test normal
        @test InfiniteOpt._update_param_var_mapping(ivref, tuple) isa Nothing
        @test InfiniteOpt._infinite_variable_dependencies(pref) == [idx]
        @test InfiniteOpt._infinite_variable_dependencies(prefs[1]) == [idx]
        @test InfiniteOpt._infinite_variable_dependencies(prefs[2]) == [idx]
        # undo changes
        empty!(InfiniteOpt._infinite_variable_dependencies(pref))
        empty!(InfiniteOpt._infinite_variable_dependencies(prefs[1]))
        empty!(InfiniteOpt._infinite_variable_dependencies(prefs[2]))
    end
    # _check_and_make_variable_ref
    @testset "_check_and_make_variable_ref" begin
        # prepare secondary model and parameter and variable
        m2 = InfiniteModel()
        @independent_parameter(m2, pref3 in [0, 1])
        v = build_variable(error, info, Infinite, parameter_refs = pref3)
        # test for error of invalid variable
        @test_throws VariableNotOwned{GeneralVariableRef} InfiniteOpt._check_and_make_variable_ref(m, v)
        # test normal
        idx = InfiniteVariableIndex(1)
        vref = InfiniteVariableRef(m2, idx)
        @test InfiniteOpt._check_and_make_variable_ref(m2, v) == vref
        @test InfiniteOpt._infinite_variable_dependencies(pref3) == [idx]
        # test with other variable object
        @test_throws ArgumentError InfiniteOpt._check_and_make_variable_ref(m, :bad)
    end
    # add_variable
    @testset "JuMP.add_variable" begin
        # prepare secondary model and parameter and variable
        m2 = InfiniteModel()
        @independent_parameter(m2, pref3 in [0, 1])
        v = build_variable(error, info, Infinite, parameter_refs = pref3)
        # test for error of invalid variable
        @test_throws VariableNotOwned{GeneralVariableRef} add_variable(m, v)
        # prepare normal variable
        v = build_variable(error, info, Infinite, parameter_refs = pref)
        # test normal
        idx = InfiniteVariableIndex(1)
        vref = InfiniteVariableRef(m, idx)
        gvref = InfiniteOpt._make_variable_ref(m, idx)
        @test add_variable(m, v, "name") == gvref
        @test haskey(InfiniteOpt._data_dictionary(vref), idx)
        @test InfiniteOpt._core_variable_object(vref) == v
        @test InfiniteOpt._infinite_variable_dependencies(pref) == [idx]
        @test name(vref) == "name(pref)"
        # prepare infinite variable with all the possible info additions
        v = build_variable(error, info2, Infinite, parameter_refs = pref)
        # test info addition functions
        idx = InfiniteVariableIndex(2)
        vref = InfiniteVariableRef(m, idx)
        gvref = InfiniteOpt._make_variable_ref(m, idx)
        @test add_variable(m, v, "name") == gvref
        @test !optimizer_model_ready(m)
        # lower bound
        cindex = ConstraintIndex(1)
        cref = InfiniteConstraintRef(m, cindex, ScalarShape())
        @test has_lower_bound(vref)
        @test JuMP._lower_bound_index(vref) == cindex
        @test constraint_object(cref) isa ScalarConstraint{GeneralVariableRef,
                                                           MOI.GreaterThan{Float64}}
        @test InfiniteOpt._data_object(cref).is_info_constraint
        # upper bound
        cindex = ConstraintIndex(2)
        cref = InfiniteConstraintRef(m, cindex, ScalarShape())
        @test has_upper_bound(vref)
        @test JuMP._upper_bound_index(vref) == cindex
        @test constraint_object(cref) isa ScalarConstraint{GeneralVariableRef,
                                                           MOI.LessThan{Float64}}
        @test InfiniteOpt._data_object(cref).is_info_constraint
        # fix
        cindex = ConstraintIndex(3)
        cref = InfiniteConstraintRef(m, cindex, ScalarShape())
        @test is_fixed(vref)
        @test JuMP._fix_index(vref) == cindex
        @test constraint_object(cref) isa ScalarConstraint{GeneralVariableRef,
                                                           MOI.EqualTo{Float64}}
        @test InfiniteOpt._data_object(cref).is_info_constraint
        # binary
        cindex = ConstraintIndex(4)
        cref = InfiniteConstraintRef(m, cindex, ScalarShape())
        @test is_binary(vref)
        @test JuMP._binary_index(vref) == cindex
        @test constraint_object(cref) isa ScalarConstraint{GeneralVariableRef,
                                                           MOI.ZeroOne}
        @test InfiniteOpt._data_object(cref).is_info_constraint
        @test InfiniteOpt._constraint_dependencies(vref) == [ConstraintIndex(i)
                                                             for i = 1:4]
        # prepare infinite variable with integer info addition
        v = build_variable(error, info3, Infinite, parameter_refs = pref)
        # test integer addition functions
        idx = InfiniteVariableIndex(3)
        vref = InfiniteVariableRef(m, idx)
        gvref = InfiniteOpt._make_variable_ref(m, idx)
        @test add_variable(m, v, "name") == gvref
        @test !optimizer_model_ready(m)
        cindex = ConstraintIndex(8)
        cref = InfiniteConstraintRef(m, cindex, ScalarShape())
        @test is_integer(vref)
        @test JuMP._integer_index(vref) == cindex
        @test constraint_object(cref) isa ScalarConstraint{GeneralVariableRef,
                                                           MOI.Integer}
        @test InfiniteOpt._data_object(cref).is_info_constraint
        @test InfiniteOpt._constraint_dependencies(vref) == [ConstraintIndex(i)
                                                             for i = 5:8]
    end
end

# Test helper functions for infinite variable macro
@testset "Macro Helpers" begin
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
@testset "Macro Definition" begin
    # initialize model and infinite parameters
    m = InfiniteModel()
    @independent_parameter(m, t in [0, 1])
    @dependent_parameters(m, x[1:2] in [-1, 1])
    # test single variable definition
    @testset "Single" begin
        # test basic defaults
        idx = InfiniteVariableIndex(1)
        vref = InfiniteVariableRef(m, idx)
        gvref = InfiniteOpt._make_variable_ref(m, idx)
        @test @infinite_variable(m, parameter_refs = t) == gvref
        @test name(vref) == "noname(t)"
        # test more tuple input and variable details
        idx = InfiniteVariableIndex(2)
        vref = InfiniteVariableRef(m, idx)
        gvref = InfiniteOpt._make_variable_ref(m, idx)
        @test @infinite_variable(m, parameter_refs = (t, x), base_name = "test",
                                 binary = true) == gvref
        @test name(vref) == "test(t, x)"
        @test is_binary(vref)
        # test nonanonymous with simple single arg
        idx = InfiniteVariableIndex(3)
        vref = InfiniteVariableRef(m, idx)
        gvref = InfiniteOpt._make_variable_ref(m, idx)
        @test @infinite_variable(m, a(x)) == gvref
        @test name(vref) == "a(x)"
        # test nonanonymous with complex single arg
        idx = InfiniteVariableIndex(4)
        vref = InfiniteVariableRef(m, idx)
        gvref = InfiniteOpt._make_variable_ref(m, idx)
        @test @infinite_variable(m, 0 <= b(x) <= 1) == gvref
        @test name(vref) == "b(x)"
        @test lower_bound(vref) == 0
        @test upper_bound(vref) == 1
        # test nonanonymous with reversed single arg
        idx = InfiniteVariableIndex(5)
        vref = InfiniteVariableRef(m, idx)
        gvref = InfiniteOpt._make_variable_ref(m, idx)
        @test @infinite_variable(m, 0 <= c(t)) == gvref
        @test name(vref) == "c(t)"
        @test lower_bound(vref) == 0
        # test multi-argument expr 1
        idx = InfiniteVariableIndex(6)
        vref = InfiniteVariableRef(m, idx)
        gvref = InfiniteOpt._make_variable_ref(m, idx)
        @test @infinite_variable(m, d(t) == 0, Int, base_name = "test") == gvref
        @test name(vref) == "test(t)"
        @test fix_value(vref) == 0
    end
    # test array variable definition
    @testset "Array" begin
        # test anonymous array
        idxs = [InfiniteVariableIndex(7), InfiniteVariableIndex(8)]
        vrefs = [InfiniteVariableRef(m, idx) for idx in idxs]
        gvrefs = [InfiniteOpt._make_variable_ref(m, idx) for idx in idxs]
        @test @infinite_variable(m, [1:2], parameter_refs = t) == gvrefs
        @test name(vrefs[1]) == "noname(t)"
        # test basic param expression
        idxs = [InfiniteVariableIndex(9), InfiniteVariableIndex(10)]
        vrefs = [InfiniteVariableRef(m, idx) for idx in idxs]
        gvrefs = [InfiniteOpt._make_variable_ref(m, idx) for idx in idxs]
        @test @infinite_variable(m, e[1:2], parameter_refs = (t, x)) == gvrefs
        @test name(vrefs[2]) == "e[2](t, x)"
        # test comparison without params
        idxs = [InfiniteVariableIndex(11), InfiniteVariableIndex(12)]
        vrefs = [InfiniteVariableRef(m, idx) for idx in idxs]
        gvrefs = [InfiniteOpt._make_variable_ref(m, idx) for idx in idxs]
        @test @infinite_variable(m, 0 <= f[1:2] <= 1,
                                 parameter_refs = (t, x)) == gvrefs
        @test name(vrefs[2]) == "f[2](t, x)"
        @test lower_bound(vrefs[1]) == 0
        @test upper_bound(vrefs[2]) == 1
        # test comparison with call
        idxs = [InfiniteVariableIndex(13), InfiniteVariableIndex(14)]
        vrefs = [InfiniteVariableRef(m, idx) for idx in idxs]
        gvrefs = [InfiniteOpt._make_variable_ref(m, idx) for idx in idxs]
        @test @infinite_variable(m, 0 <= g[1:2](t) <= 1) == gvrefs
        @test name(vrefs[1]) == "g[1](t)"
        @test lower_bound(vrefs[1]) == 0
        @test upper_bound(vrefs[2]) == 1
        # test fixed
        idxs = [InfiniteVariableIndex(15), InfiniteVariableIndex(16)]
        vrefs = [InfiniteVariableRef(m, idx) for idx in idxs]
        gvrefs = [InfiniteOpt._make_variable_ref(m, idx) for idx in idxs]
        @test @infinite_variable(m, h[i = 1:2](t) == ones(2)[i]) == gvrefs
        @test name(vrefs[1]) == "h[1](t)"
        @test fix_value(vrefs[1]) == 1
        # test containers
        idxs = [InfiniteVariableIndex(17), InfiniteVariableIndex(18)]
        vrefs = [InfiniteVariableRef(m, idx) for idx in idxs]
        gvrefs = [InfiniteOpt._make_variable_ref(m, idx) for idx in idxs]
        svrefs = convert(JuMP.Containers.SparseAxisArray, gvrefs)
        @test @infinite_variable(m, [1:2](t),
                                 container = SparseAxisArray) isa JuMPC.SparseAxisArray
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
        gvref = InfiniteOpt._make_variable_ref(m, InfiniteVariableIndex(1))
        @test_macro_throws ErrorException @infinite_variable(m, i(t),
                                                  infinite_variable_ref = gvref)
        @test_macro_throws ErrorException @infinite_variable(m, i(t), bad = 42)
        # test name duplication
        @test_macro_throws ErrorException @infinite_variable(m, a(t), Int)
    end
end

# test usage methods
@testset "Usage" begin
    # initialize model and stuff
    m = InfiniteModel()
    @independent_parameter(m, t in [0, 1])
    @dependent_parameters(m, x[1:2] in [-1, 1])
    @infinite_variable(m, y(t, x))
    vref = dispatch_variable_ref(y)
    # test used_by_reduced_variable
    @testset "used_by_reduced_variable" begin
        @test !used_by_reduced_variable(vref)
        push!(InfiniteOpt._reduced_variable_dependencies(vref),
              ReducedInfiniteVariableIndex(1))
        @test used_by_reduced_variable(y)
        @test used_by_reduced_variable(vref)
        empty!(InfiniteOpt._reduced_variable_dependencies(vref))
    end
    # test used_by_point_variable
    @testset "used_by_point_variable" begin
        @test !used_by_point_variable(vref)
        push!(InfiniteOpt._point_variable_dependencies(vref), PointVariableIndex(1))
        @test used_by_point_variable(y)
        @test used_by_point_variable(vref)
        empty!(InfiniteOpt._point_variable_dependencies(vref))
    end
    # test used_by_measure
    @testset "used_by_measure" begin
        @test !used_by_measure(vref)
        push!(InfiniteOpt._measure_dependencies(vref), MeasureIndex(1))
        @test used_by_measure(y)
        @test used_by_measure(vref)
        empty!(InfiniteOpt._measure_dependencies(vref))
    end
    # test used_by_constraint
    @testset "used_by_constraint" begin
        @test !used_by_constraint(vref)
        push!(InfiniteOpt._constraint_dependencies(vref), ConstraintIndex(1))
        @test used_by_constraint(y)
        @test used_by_constraint(vref)
        empty!(InfiniteOpt._constraint_dependencies(vref))
    end
    # test used_by_objective
    @testset "used_by_objective" begin
        @test !used_by_objective(y)
        @test !used_by_objective(vref)
    end
    # test is_used
    @testset "is_used" begin
        # test not used
        @test !is_used(vref)
        # test used by constraint and/or measure
        push!(InfiniteOpt._constraint_dependencies(vref), ConstraintIndex(1))
        @test is_used(y)
        empty!(InfiniteOpt._constraint_dependencies(vref))
        # test used by point variable
        num = Float64(0)
        info = VariableInfo(false, num, false, num, false, num, false, num, false, false)
        var = PointVariable(info, y, [0., 0., 0.])
        object = VariableData(var, "var")
        idx = PointVariableIndex(1)
        pvref = PointVariableRef(m, idx)
        @test InfiniteOpt._add_data_object(m, object) == idx
        push!(InfiniteOpt._point_variable_dependencies(vref), idx)
        @test !is_used(vref)
        push!(InfiniteOpt._constraint_dependencies(pvref), ConstraintIndex(2))
        @test is_used(vref)
        empty!(InfiniteOpt._point_variable_dependencies(vref))
        # test used by reduced variable
        eval_supps = Dict{Int, Float64}(1 => 0.5, 3 => 1)
        var = ReducedInfiniteVariable(y, eval_supps, [2])
        object = VariableData(var, "var")
        idx = ReducedInfiniteVariableIndex(1)
        rvref = ReducedInfiniteVariableRef(m, idx)
        @test InfiniteOpt._add_data_object(m, object) == idx
        push!(InfiniteOpt._reduced_variable_dependencies(vref), idx)
        @test !is_used(vref)
        push!(InfiniteOpt._constraint_dependencies(rvref), ConstraintIndex(2))
        @test is_used(vref)
    end
end

# Test queries for parameter references and values
@testset "Parameter Modification" begin
    # initialize model, parameter, and variables
    m = InfiniteModel()
    @independent_parameter(m, pref in [0, 1])
    @independent_parameter(m, pref2 in [0, 1])
    @dependent_parameters(m, prefs[1:2] in [0, 1])
    @finite_parameter(m, fin, 42)
    @infinite_variable(m, ivref(pref, pref2, prefs) == 1)
    dvref = dispatch_variable_ref(ivref)
    # _update_variable_param_refs
    @testset "_update_variable_param_refs" begin
        orig_prefs = raw_parameter_refs(dvref)
        @test isa(InfiniteOpt._update_variable_param_refs(dvref, VectorTuple(pref2)),
                  Nothing)
        @test parameter_refs(dvref) == (pref2, )
        @test name(dvref) == "ivref(pref2)"
        @test isa(InfiniteOpt._update_variable_param_refs(dvref, orig_prefs),
                  Nothing)
    end
    # set_parameter_refs
    @testset "set_parameter_refs" begin
        # test normal with 1 param
        @test isa(set_parameter_refs(ivref, (pref2, )), Nothing)
        @test parameter_refs(dvref) == (pref2, )
        @test name(dvref) == "ivref(pref2)"
        # test double specify
        @test_throws ErrorException set_parameter_refs(dvref, (pref2, pref2))
        # test used by point variable
        push!(InfiniteOpt._point_variable_dependencies(dvref),
              PointVariableIndex(1))
        @test_throws ErrorException set_parameter_refs(dvref, (pref, ))
        empty!(InfiniteOpt._point_variable_dependencies(dvref))
        # test used by reduced variable
        push!(InfiniteOpt._reduced_variable_dependencies(dvref),
              ReducedInfiniteVariableIndex(1))
        @test_throws ErrorException set_parameter_refs(dvref, (pref, ))
        empty!(InfiniteOpt._reduced_variable_dependencies(dvref))
    end
    # add_parameter_ref
    @testset "add_parameter_ref" begin
        # test used by point variable
        push!(InfiniteOpt._point_variable_dependencies(dvref),
              PointVariableIndex(1))
        @test_throws ErrorException add_parameter_ref(ivref, pref)
        empty!(InfiniteOpt._point_variable_dependencies(dvref))
        # test used by reduced variable
        push!(InfiniteOpt._reduced_variable_dependencies(dvref),
              ReducedInfiniteVariableIndex(1))
        @test_throws ErrorException add_parameter_ref(dvref, pref)
        empty!(InfiniteOpt._reduced_variable_dependencies(dvref))
        # test normal use
        @test isa(add_parameter_ref(ivref, pref), Nothing)
        @test parameter_refs(ivref) == (pref2, pref)
        @test name(ivref) == "ivref(pref2, pref)"
        # test duplication error
        @test_throws ErrorException add_parameter_ref(dvref, pref2)
        #test bad array error
        @test_throws ErrorException add_parameter_ref(dvref, prefs[1:1])
    end
end
