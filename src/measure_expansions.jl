"""
    make_point_variable_ref(ivref::InfiniteVariableRef)::PointVariableRef

Internal method for measure expansion that returns a `PointVariableRef` with
a unique index. This is useful for writing extensions to [`expand_measure`](@ref).
"""
function make_point_variable_ref(ivref::InfiniteVariableRef)::PointVariableRef
    inf_model = JuMP.owner_model(ivref)
    index = inf_model.next_var_index += 1
    return PointVariableRef(inf_model, index)
end

## Helper function for making place holder infinite variables
"""
    make_reduced_variable_ref(ivref::InfiniteVariableRef, removed_index::Int,
                              support::Union{Number,
                              JuMPC.SparseAxisArray{<:Number}}
                              )::ReducedInfiniteVariableRef

Internal method for measure expansion that returns a `ReducedInfiniteVariableRef`
and that is connected with `ivref.model`. This method is intended for intial
reduction of an infinite variable. Here `removed_index` is the index of the
reduced parameter in `parameter_refs(ivref)` and `support` is the value that will
support that particular infinite parameter. This is useful for writing
extensions to [`expand_measure`](@ref).
"""
function make_reduced_variable_ref(ivref::InfiniteVariableRef, removed_index::Int,
                                support::Union{Number,
                                JuMPC.SparseAxisArray{<:Number}}
                                )::ReducedInfiniteVariableRef
    inf_model = JuMP.owner_model(ivref)
    index = inf_model.next_var_index += 1
    inf_model.reduced_variable[index] = ReducedInfiniteVariable(ivref,
                                                 Dict(removed_index => support))
    if haskey(inf_model.infinite_to_reduced, JuMP.index(ivref))
        push!(inf_model.infinite_to_reduced[JuMP.index(ivref)], index)
    else
        inf_model.infinite_to_reduced[JuMP.index(ivref)] = [index]
    end
    return ReducedInfiniteVariableRef(inf_model, index)
end

"""
    make_reduced_variable_ref(ivref::InfiniteVariableRef,
                              supports::Dict)::ReducedInfiniteVariableRef

Internal method for measure expansion that returns a `ReducedInfiniteVariableRef`
and that is connected with `ivref.model`. This method is intended for further
reduction of an infinite variable based on an existing reduced infinite variable.
Here `supports` is a dictionary mapping the indices of parameters in
`parameter_refs(ivref)` to their supports values. This is useful for writing
extensions to [`expand_measure`](@ref).
"""
function make_reduced_variable_ref(ivref::InfiniteVariableRef,
                                    supports::Dict)::ReducedInfiniteVariableRef
    inf_model = JuMP.owner_model(ivref)
    index = inf_model.next_var_index += 1
    inf_model.reduced_variable[index] = ReducedInfiniteVariable(ivref, copy(supports))
    if haskey(inf_model.infinite_to_reduced, JuMP.index(ivref))
        push!(inf_model.infinite_to_reduced[JuMP.index(ivref)], index)
    else
        inf_model.infinite_to_reduced[JuMP.index(ivref)] = [index]
    end
    return ReducedInfiniteVariableRef(inf_model, index)
end

## Make helper functions for extracting evaluated parameter values
# DiscreteMeasureData
function _get_param_value_list(pref::ParameterRef, data::DiscreteMeasureData)
    return supports(data)
end

# MultiDiscreteMeasureData
function _get_param_value_list(pref::ParameterRef,
                               data::MultiDiscreteMeasureData)
    key = first(filter(p -> p[2] == pref, parameter_refs(data).data))[1]
    return [supports(data)[i][key] for i in eachindex(supports(data))]
end


"""
    expand_measure(expr, data::AbstractMeasureData,
                   write_model::JuMP.AbstractModel,
                   point_mapper::Function)::JuMP.AbstractJuMPScalar

Return the finite reformulation of a measure containing a variable/parameter
expression `expr` with measure data `data`. Here `write_model` is the target
model where this expanded expression will be used. Thus, any variables that need
to be created will be added to `write_model`. The function `point_mapper` should
be called to make and map point variables via the syntax:
`point_mapper(write_model, pvref, ivref, support)`. Here `ivref` denotes the
`InfiniteVariableRef` that the point variable will be based on at `support` where
`support` is a `Tuple` of values that matches the format of `parameter_refs(ivref)`.
`pvref` is the pre-made `ParameterRef` that should be created via
[`make_point_variable_ref`](@ref). [`make_reduced_variable_ref`](@ref) may also
be helpful. Note this is intended as an internal function,
but will need to be extended for unsupported `expr` types and for user-defined
measure data types.
"""
function expand_measure end

# TODO fix reduced variables to address case of partial vector indexing

# InfiniteVariableRef
function expand_measure(ivref::InfiniteVariableRef,
                         data::Union{DiscreteMeasureData,
                                     MultiDiscreteMeasureData},
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    # figure out the parameter groups
    group = group_id(first(data.parameter_ref))
    groups = _group.(parameter_refs(ivref))
    # prepare return AffExpr and get necessary information
    aff = zero(JuMP.GenericAffExpr{Float64, GeneralVariableRef})
    # treat variable as constant if doesn't have measure parameter
    if !(group in groups)
        for i in eachindex(data.supports)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    ivref)
        end
    # convert variable into point variables if its only parameter is the
    # measure parameter
    elseif length(parameter_refs(ivref)) == 1
        for i in eachindex(data.supports)
            pvref = make_point_variable_ref(ivref)
            point_mapper(trans_model, pvref, ivref, (data.supports[i],))
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    pvref)
        end
    # make reduced variables if the variable contains other parameters
    else
        tuple_loc = findfirst(isequal(group), groups)
        for i in eachindex(data.supports)
            rvref = make_reduced_variable_ref(ivref, tuple_loc, data.supports[i])
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    rvref)
        end
    end
    return aff
end

# ReducedInfiniteVariableRef
function expand_measure(rvref::ReducedInfiniteVariableRef,
                         data::Union{DiscreteMeasureData,
                                     MultiDiscreteMeasureData},
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    # figure out the parameters used by the reduced infinite variable
    orig_prefs = parameter_refs(infinite_variable_ref(rvref))
    # figure out the parameter groups
    group = group_id(first(data.parameter_ref))
    groups = _group.(parameter_refs(rvref))
    # prepare return AffExpr and get necessary information
    aff = zero(JuMP.GenericAffExpr{Float64, GeneralVariableRef})
    # treat variable as constant if doesn't have measure parameter
    if !(group in groups)
        for i in eachindex(data.supports)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    rvref)
        end
    # convert variable into point variables if its only parameter is the
    # measure parameter
    elseif length(parameter_refs(rvref)) == 1
        tuple_loc = findfirst(isequal(group), _group.(orig_prefs))
        for i in eachindex(data.supports)
            pvref = make_point_variable_ref(infinite_variable_ref(rvref))
            _reduced_variable(rvref).eval_supports[tuple_loc] = data.supports[i]
            support = Tuple(eval_supports(rvref)[j] for j in eachindex(eval_supports(rvref)))
            point_mapper(trans_model, pvref, infinite_variable_ref(rvref), support)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    pvref)
        end
    # make reduced variables if the variable contains other parameters
    else
        tuple_loc = findfirst(isequal(group), _group.(orig_prefs))
        for i in eachindex(data.supports)
            new_rvref = make_reduced_variable_ref(infinite_variable_ref(rvref),
                                               eval_supports(rvref))
            _reduced_variable(new_rvref).eval_supports[tuple_loc] = data.supports[i]
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    new_rvref)
        end
    end
    return aff
end

# FiniteVariableRef
function expand_measure(vref::FiniteVariableRef,
                         data::Union{DiscreteMeasureData,
                                     MultiDiscreteMeasureData},
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    aff = zero(JuMP.GenericAffExpr{Float64, GeneralVariableRef})
    # treat the variable as a constant
    for i in eachindex(data.supports)
        JuMP.add_to_expression!(aff, data.coefficients[i] *
                                data.weight_function(data.supports[i]),
                                vref)
    end
    return aff
end

# ParameterRef with scalar data
function expand_measure(pref::ParameterRef,
                         data::DiscreteMeasureData,
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    aff = zero(JuMP.GenericAffExpr{Float64, GeneralVariableRef})
    # replace the parameter with its value if it is the measure parameter
    if data.parameter_ref == pref
        for i in eachindex(data.supports)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]) *
                                    data.supports[i])
        end
    # treat the parameter as a constant otherwise
    else
        for i in eachindex(data.supports)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    pref)
        end
    end
    return aff
end

# ParameterRef with vector data
function expand_measure(pref::ParameterRef,
                         data::MultiDiscreteMeasureData,
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    aff = zero(JuMP.GenericAffExpr{Float64, GeneralVariableRef})
    # determine if pref is part of the measure parameters
    pref_dict = filter(p -> p[2] == pref, data.parameter_ref.data)
    # replace the parameter with its value if it is the measure parameter
    if length(pref_dict) != 0
        for i in eachindex(data.supports)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]) *
                                    data.supports[i][collect(keys(pref_dict))[1]])
        end
    # treat the parameter as a constant otherwise
    else
        for i in eachindex(data.supports)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    pref)
        end
    end
    return aff
end

# GenericAffExpr
function expand_measure(expr::JuMP.GenericAffExpr,
                         data::Union{DiscreteMeasureData,
                                     MultiDiscreteMeasureData},
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    # need to use a quadratic expression in case contains measures with
    # quadratic expressions
    quad = zero(JuMP.GenericQuadExpr{Float64, GeneralVariableRef})
    # expand each variable independently and add all together
    for (var, coef) in expr.terms
        JuMP.add_to_expression!(quad, coef, expand_measure(var, data,
                                                            trans_model,
                                                            point_mapper))
    end
    # expand over the constant
    if expr.constant != 0
        for i in eachindex(data.supports)
            JuMP.add_to_expression!(quad, data.coefficients[i] *
                                    data.weight_function(data.supports[i]) *
                                    expr.constant)
        end
    end
    # return affexpr if appropriate
    if length(quad.terms) == 0
        return quad.aff
    else
        return quad
    end
end

# GenericQuadExpr
function expand_measure(expr::JuMP.GenericQuadExpr,
                         data::Union{DiscreteMeasureData,
                                     MultiDiscreteMeasureData},
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    quad = zero(JuMP.GenericQuadExpr{Float64, GeneralVariableRef})
    # convert the GenericAffExpr
    quad.aff = expand_measure(expr.aff, data, trans_model, point_mapper)
    for (pair, coef) in expr.terms
        # expand on both variables
        expr_a = expand_measure(pair.a, data, trans_model, point_mapper)
        expr_b = expand_measure(pair.b, data, trans_model, point_mapper)
        vars_a = collect(keys(expr_a.terms))
        vars_b = collect(keys(expr_b.terms))
        alt_terms = false
        # check for case that a variable was a parameter converted to a number
        if length(vars_a) == 0
            vars_a = _get_param_value_list(pair.a, data)
            terms = [data.coefficients[i] * data.weight_function(data.supports[i]) for i in eachindex(data.supports)]
            alt_terms = true
        end
        if length(vars_b) == 0
            vars_b = _get_param_value_list(pair.b, data)
            terms = [data.coefficients[i] * data.weight_function(data.supports[i]) for i in eachindex(data.supports)]
            alt_terms = true
        end
        # combine both variable expressions using the coefficients from one
        # of them
        if length(vars_a) == length(vars_b)
            # are same length therefore have same coefficients
            for i in eachindex(vars_a)
                if alt_terms
                    JuMP.add_to_expression!(quad, coef * terms[i], vars_a[i],
                                            vars_b[i])
                else
                    JuMP.add_to_expression!(quad, coef * expr_a.terms[vars_a[i]],
                                            vars_a[i], vars_b[i])
                end
            end
        elseif length(vars_a) == 1
            # var_a was effectively a constant and var_b was't
            for i in eachindex(vars_b)
                if alt_terms
                    JuMP.add_to_expression!(quad, coef * terms[i], vars_a[1],
                                            vars_b[i])
                else
                    JuMP.add_to_expression!(quad, coef * expr_b.terms[vars_b[i]],
                                            vars_a[1], vars_b[i])
                end
            end
        else
            # var_b was effectively a constant and var_a was't
            for i in eachindex(vars_a)
                if alt_terms
                    JuMP.add_to_expression!(quad, coef * terms[i], vars_a[i],
                                            vars_b[1])
                else
                    JuMP.add_to_expression!(quad, coef * expr_a.terms[vars_a[i]],
                                            vars_a[i], vars_b[1])
                end
            end
        end
    end
    return quad
end

# MeasureRef
function expand_measure(mref::MeasureRef,
                         data::Union{DiscreteMeasureData,
                                     MultiDiscreteMeasureData},
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    # determine function and data of the inner measure
    deeper_func = measure_function(mref)
    deeper_data = measure_data(mref)
    # expand the inner measure (note this is recursive for nested measures)
    new_func = expand_measure(deeper_func, deeper_data, trans_model,
                               point_mapper)
    # expand current level with the inner measure now expanded
    return expand_measure(new_func, data, trans_model, point_mapper)
end

# Catch all method for undefined behavior
function expand_measure(expr, data::AbstractMeasureData,
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)
    expr_type = typeof(expr)
    data_type = typeof(data)
    error("Undefined behavior to expand expression of type `$expr_type` with " *
          "measure data `$data_type`. If this functionality is needed consider " *
          "extending `expand_measure`.")
end

# Map temp point variable references to actual variables by adding a variable to
# the infinite model. This is used by expand(mref) to expand measures within an
# infinite model
function _add_mapped_point_variable(model::InfiniteModel,
                                    pvref::PointVariableRef,
                                    ivref::InfiniteVariableRef, support::Tuple)
    # build variable
    var = JuMP.build_variable(error, _variable_info(ivref), Point,
                              infinite_variable_ref = ivref,
                              parameter_values = support)
    # add the variable completely (note the reference is already made)
    desired_index = JuMP.index(pvref)
    curr_index = model.next_var_index
    model.next_var_index = desired_index - 1
    JuMP.add_variable(model, var)
    model.next_var_index = curr_index
    return
end

"""
    expand(mref::MeasureRef)::JuMP.AbstractJuMPScalar

Return a JuMP scalar function containing the explicit expansion of the measure
`mref`. This expansion is done according to the measure data. Note that
variables are added to the model as necessary to accomodate the expansion (i.e.,
point variables and reduced infinite variables are made as needed). Errors if
expansion is undefined for the measure data and/or the measure expression. If
desired this can be used in combination with [`measure`](@ref) to expand measures
on the fly.

This is useful for extensions that employ a custom optimizer model since it
can be used evaluate measures before expressions are translated to the new model.
This method can also be extended to handle custom measure data types by extending
```expand_measure``` which should be of the form:
```julia
expand_measure(expr::AbstractJuMPScalar, data::AbstractMeasureData,
                model::JuMP.AbstractModel, point_mapper::Function)
```
See [`expand_measure`](@ref) for more information.

**Example**
```julia-repl
julia> tdata = DiscreteMeasureData(t, [0.5, 0.5], [0, 1])

julia> expr = expand(measure(g + z + T - h - 2, tdata))
0.5 g(0) + 0.5 g(1) + z + 0.5 T(0, x) + 0.5 T(1, x) - h(x) - 2
```
"""
function expand(mref::MeasureRef)::JuMP.AbstractJuMPScalar
    return expand_measure(measure_function(mref), measure_data(mref),
                           JuMP.owner_model(mref), _add_mapped_point_variable)
end

## Helper functions for expanding the measure references in expressions
# MeasureRef
function _expand_measures(mref::MeasureRef,
                          expand_model::JuMP.AbstractModel,
                          point_mapper::Function)::JuMP.AbstractJuMPScalar
    return expand_measure(measure_function(mref), measure_data(mref),
                           expand_model, point_mapper)
end

# GenericAffExpr
function _expand_measures(expr::JuMP.GenericAffExpr{C, <:GeneralVariableRef},
                          expand_model::JuMP.AbstractModel,
                          point_mapper::Function)::JuMP.AbstractJuMPScalar where {C}
    # use a QuadExpr in case measures contain quadratic espressions
    quad = zero(JuMP.GenericQuadExpr{C, GeneralVariableRef})
    quad.aff.constant = expr.constant
    # add the variables to the expr, converting measures into expanded exprs
    for (var, coef) in expr.terms
        if isa(var, MeasureRef)
            JuMP.add_to_expression!(quad, coef, expand_measure(measure_function(var),
                                                                measure_data(var),
                                                                expand_model,
                                                                point_mapper))
        else
            JuMP.add_to_expression!(quad, coef, var)
        end
    end
    # return a AffExpr if there are no quadratic terms
    if length(quad.terms) == 0
        return quad.aff
    else
        return quad
    end
end

# GenericQuadExpr
function _expand_measures(expr::JuMP.GenericQuadExpr{C, <:GeneralVariableRef},
                          expand_model::JuMP.AbstractModel,
                          point_mapper::Function)::JuMP.GenericQuadExpr where {C}
    quad = zero(JuMP.GenericQuadExpr{C, GeneralVariableRef})
    quad.aff = _expand_measures(expr.aff, expand_model, point_mapper)
    # add the quadratic terms to the expr, converting measures into expanded exprs
    # note that this will error if the expanded forms are not linear or quadratic
    for (pair, coef) in expr.terms
        var_a = pair.a
        var_b = pair.b
        if isa(var_a, MeasureRef)
            var_a = expand_measure(measure_function(var_a), measure_data(var_a),
                                    expand_model, point_mapper)
        end
        if isa(var_b, MeasureRef)
            var_b = expand_measure(measure_function(var_b), measure_data(var_b),
                                    expand_model, point_mapper)
        end
        JuMP.add_to_expression!(quad, convert(JuMP.GenericQuadExpr{C,
                                GeneralVariableRef}, coef * var_a * var_b))
    end
    return quad
end

"""
    expand_all_measures!(model::InfiniteModel)

Expand all of the measures used in the objective and/or constraints of `model`.
The objective and constraints are updated accordingly. Note that
variables are added to the model as necessary to accomodate the expansion (i.e.,
point variables and reduced infinite variables are made as needed). Errors if
expansion is undefined for the measure data and/or the measure expression. Also
errors if the expanded objective function is not finite.

This is useful for extensions that employ a custom optimizer model since it
can be used evaluate measures before `model` is translated into the new model.
This method can also be extended to handle custom measure data types by extending
```expand_measure``` which should be of the form:
```julia
expand_measure(expr::AbstractJuMPScalar, data::AbstractMeasureData,
                model::JuMP.AbstractModel, point_mapper::Function)
```
See [`expand_measure`](@ref) for more information.

**Example**
```julia-repl
julia> print(model)
Min measure(g(t)*t) + z
Subject to
 T(t, xi) >= 0.0
 z >= 0.0
 g(t) + z >= 42.0
 measure(T(t, xi)) >= 0.0, for all xi in [-1, 1]
 t in [0, 6]
 xi in Normal(μ=0.0, σ=1.0)

julia> expand_all_measures!(model)

julia> print(model)
Min 3 g(6) + z
Subject to
 T(t, xi) >= 0.0
 z >= 0.0
 g(t) + z >= 42.0
 0.5 T(0, xi) + 0.5 T(6, xi) >= 0.0, for all xi in [-1, 1]
 t in [0, 6]
 xi in Normal(μ=0.0, σ=1.0)
```
"""
function expand_all_measures!(model::InfiniteModel)
    # expand the objective if it contains measures
    if JuMP.objective_function_type(model) <: MeasureExpr
        new_obj = _possible_convert(FiniteVariableRef,
                         _expand_measures(JuMP.objective_function(model), model,
                                          _add_mapped_point_variable))
        isa(new_obj, InfiniteExpr) && error("Objective is not finite, ensure " *
                                            "all infinite variables/parameters " *
                                            "in measures are evaluated " *
                                            "completely.")
        JuMP.set_objective_function(model, new_obj)
    end
    # expand all of the constraints that contain measures
    for cindex in sort(unique(vcat(values(model.meas_to_constrs)...)))
        # expand the expression
        new_func = _possible_convert(FiniteVariableRef,
                                 _expand_measures(model.constrs[cindex].func,
                                                  model,
                                                  _add_mapped_point_variable))
        # get the necessary info
        cref = _make_constraint_ref(model, cindex)
        name = JuMP.name(cref)
        set = model.constrs[cindex].set
        curr_index = model.next_constr_index
        # delete the old cosntraint and replace it with the expanded version
        model.next_constr_index = cindex - 1
        if isa(model.constrs[cindex], BoundedScalarConstraint) && isa(new_func,
                                                                   InfiniteExpr)
            orig_bounds = model.constrs[cindex].orig_bounds
            JuMP.delete(model, cref)
            JuMP.add_constraint(model, JuMP.build_constraint(error, new_func,
                                set; parameter_bounds = orig_bounds), name)
        else
            JuMP.delete(model, cref)
            JuMP.add_constraint(model, JuMP.build_constraint(error, new_func,
                                set), name)
        end
        model.next_constr_index = curr_index
    end
    return
end
