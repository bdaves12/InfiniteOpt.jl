# Helper function for making place holder point variables
function _make_point_variable(ivref::InfOptVariableRef)::InfOptVariableRef
    inf_model = JuMP.owner_model(ivref)
    index = inf_model.next_var_index += 1
    return InfOptVariableRef(inf_model, index, Point)
end

## Helper function for making place holder infinite variables
# first time reduction
function _make_reduced_variable(ivref::InfOptVariableRef, removed_index::Int,
                                support::Union{Number,
                                JuMPC.SparseAxisArray{<:Number}}
                                )::InfOptVariableRef
    inf_model = JuMP.owner_model(ivref)
    index = inf_model.next_var_index += 1
    inf_model.reduced_info[index] = ReducedInfiniteInfo(ivref,
                                                 Dict(removed_index => support))
    if haskey(inf_model.infinite_to_reduced, JuMP.index(ivref))
        push!(inf_model.infinite_to_reduced[JuMP.index(ivref)], index)
    else
        inf_model.infinite_to_reduced[JuMP.index(ivref)] = [index]
    end
    return InfOptVariableRef(inf_model, index, Reduced)
end

# further reduce
function _make_reduced_variable(ivref::InfOptVariableRef,
                                supports::Dict)::InfOptVariableRef
    inf_model = JuMP.owner_model(ivref)
    index = inf_model.next_var_index += 1
    inf_model.reduced_info[index] = ReducedInfiniteInfo(ivref, copy(supports))
    if haskey(inf_model.infinite_to_reduced, JuMP.index(ivref))
        push!(inf_model.infinite_to_reduced[JuMP.index(ivref)], index)
    else
        inf_model.infinite_to_reduced[JuMP.index(ivref)] = [index]
    end
    return InfOptVariableRef(inf_model, index, Reduced)
end

## Make helper functions for extracting evaluated parameter values
# DiscreteMeasureData
function _get_param_value_list(pref::InfOptVariableRef, data::DiscreteMeasureData)
    return data.supports
end

# MultiDiscreteMeasureData
function _get_param_value_list(pref::InfOptVariableRef,
                               data::MultiDiscreteMeasureData)
    key = first(filter(p -> p[2] == pref, data.parameter_ref.data))[1]
    return [data.supports[i][key] for i = 1:length(data.supports)]
end

## Implement functions for expanding measures into regular expressions
# Function wrapper for _expand_measure
function _expand_measure(vref::InfOptVariableRef,
                         data::Union{DiscreteMeasureData,
                                     MultiDiscreteMeasureData},
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    return _expand_measure(vref, Val(variable_type(vref)), data,
                           trans_model, point_mapper)
end

# Infinite variable reference
function _expand_measure(ivref::InfOptVariableRef, ::Val{Infinite},
                         data::Union{DiscreteMeasureData,
                                     MultiDiscreteMeasureData},
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    # figure out the parameter groups
    group = group_id(first(data.parameter_ref))
    groups = _group.(parameter_refs(ivref))
    # prepare return AffExpr and get necessary information
    aff = zero(JuMP.GenericAffExpr{Float64, InfOptVariableRef})
    # treat variable as constant if doesn't have measure parameter
    if !(group in groups)
        for i = 1:length(data.supports)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    ivref)
        end
    # convert variable into point variables if its only parameter is the
    # measure parameter
    elseif length(parameter_refs(ivref)) == 1
        for i = 1:length(data.supports)
            pvref = _make_point_variable(ivref)
            point_mapper(trans_model, pvref, ivref, (data.supports[i],))
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    pvref)
        end
    # make reduced variables if the variable contains other parameters
    else
        tuple_loc = findfirst(isequal(group), groups)
        for i = 1:length(data.supports)
            rvref = _make_reduced_variable(ivref, tuple_loc, data.supports[i])
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    rvref)
        end
    end
    return aff
end

# Reduced variable reference
function _expand_measure(rvref::InfOptVariableRef, ::Val{Reduced},
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
    aff = zero(JuMP.GenericAffExpr{Float64, InfOptVariableRef})
    # treat variable as constant if doesn't have measure parameter
    if !(group in groups)
        for i = 1:length(data.supports)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    rvref)
        end
    # convert variable into point variables if its only parameter is the
    # measure parameter
    elseif length(parameter_refs(rvref)) == 1
        tuple_loc = findfirst(isequal(group), _group.(orig_prefs))
        for i = 1:length(data.supports)
            pvref = _make_point_variable(infinite_variable_ref(rvref))
            _reduced_info(rvref).eval_supports[tuple_loc] = data.supports[i]
            support = Tuple(eval_supports(rvref)[j] for j = 1:length(eval_supports(rvref)))
            point_mapper(trans_model, pvref, infinite_variable_ref(rvref), support)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    pvref)
        end
    # make reduced variables if the variable contains other parameters
    else
        tuple_loc = findfirst(isequal(group), _group.(orig_prefs))
        for i = 1:length(data.supports)
            new_rvref = _make_reduced_variable(infinite_variable_ref(rvref),
                                               eval_supports(rvref))
            _reduced_info(new_rvref).eval_supports[tuple_loc] = data.supports[i]
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    new_rvref)
        end
    end
    return aff
end

# Finite variable reference
function _expand_measure(vref::InfOptVariableRef, ::Union{Val{Global}, Val{Point}},
                         data::Union{DiscreteMeasureData,
                                     MultiDiscreteMeasureData},
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    aff = zero(JuMP.GenericAffExpr{Float64, InfOptVariableRef})
    # treat the variable as a constant
    for i = 1:length(data.supports)
        JuMP.add_to_expression!(aff, data.coefficients[i] *
                                data.weight_function(data.supports[i]),
                                vref)
    end
    return aff
end

# Parameter reference with scalar data
function _expand_measure(pref::InfOptVariableRef, ::Val{Parameter},
                         data::DiscreteMeasureData,
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    aff = zero(JuMP.GenericAffExpr{Float64, InfOptVariableRef})
    # replace the parameter with its value if it is the measure parameter
    if data.parameter_ref == pref
        for i = 1:length(data.supports)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]) *
                                    data.supports[i])
        end
    # treat the parameter as a constant otherwise
    else
        for i = 1:length(data.supports)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    pref)
        end
    end
    return aff
end

# Parameter reference with vector data
function _expand_measure(pref::InfOptVariableRef, ::Val{Parameter},
                         data::MultiDiscreteMeasureData,
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    aff = zero(JuMP.GenericAffExpr{Float64, InfOptVariableRef})
    # determine if pref is part of the measure parameters
    pref_dict = filter(p -> p[2] == pref, data.parameter_ref.data)
    # replace the parameter with its value if it is the measure parameter
    if length(pref_dict) != 0
        for i = 1:length(data.supports)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]) *
                                    data.supports[i][collect(keys(pref_dict))[1]])
        end
    # treat the parameter as a constant otherwise
    else
        for i = 1:length(data.supports)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    pref)
        end
    end
    return aff
end

# GenericAffExpr
function _expand_measure(expr::JuMP.GenericAffExpr,
                         data::Union{DiscreteMeasureData,
                                     MultiDiscreteMeasureData},
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    # need to use a quadratic expression in case contains measures with
    # quadratic expressions
    quad = zero(JuMP.GenericQuadExpr{Float64, GeneralVariableRef})
    # expand each variable independently and add all together
    for (var, coef) in expr.terms
        JuMP.add_to_expression!(quad, coef, _expand_measure(var, data,
                                                            trans_model,
                                                            point_mapper))
    end
    # expand over the constant
    if expr.constant != 0
        for i = 1:length(data.supports)
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
function _expand_measure(expr::JuMP.GenericQuadExpr,
                         data::Union{DiscreteMeasureData,
                                     MultiDiscreteMeasureData},
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    quad = zero(JuMP.GenericQuadExpr{Float64, GeneralVariableRef})
    # convert the GenericAffExpr
    quad.aff = _expand_measure(expr.aff, data, trans_model, point_mapper)
    for (pair, coef) in expr.terms
        # expand on both variables
        expr_a = _expand_measure(pair.a, data, trans_model, point_mapper)
        expr_b = _expand_measure(pair.b, data, trans_model, point_mapper)
        vars_a = collect(keys(expr_a.terms))
        vars_b = collect(keys(expr_b.terms))
        alt_terms = false
        # check for case that a variable was a parameter converted to a number
        if length(vars_a) == 0
            vars_a = _get_param_value_list(pair.a, data)
            terms = [data.coefficients[i] * data.weight_function(data.supports[i]) for i = 1:length(data.supports)]
            alt_terms = true
        end
        if length(vars_b) == 0
            vars_b = _get_param_value_list(pair.b, data)
            terms = [data.coefficients[i] * data.weight_function(data.supports[i]) for i = 1:length(data.supports)]
            alt_terms = true
        end
        # combine both variable expressions using the coefficients from one
        # of them
        if length(vars_a) == length(vars_b)
            # are same length therefore have same coefficients
            for i = 1:length(vars_a)
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
            for i = 1:length(vars_b)
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
            for i = 1:length(vars_a)
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
function _expand_measure(mref::InfOptVariableRef, ::Val{MeasureRef},
                         data::Union{DiscreteMeasureData,
                                     MultiDiscreteMeasureData},
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    # determine function and data of the inner measure
    deeper_func = measure_function(mref)
    deeper_data = measure_data(mref)
    # expand the inner measure (note this is recursive for nested measures)
    new_func = _expand_measure(deeper_func, deeper_data, trans_model,
                               point_mapper)
    # expand current level with the inner measure now expanded
    return _expand_measure(new_func, data, trans_model, point_mapper)
end

# Catch all method for undefined behavior
function _expand_measure(expr, data, trans_model::JuMP.AbstractModel,
                         point_mapper::Function)
    expr_type = typeof(expr)
    data_type = typeof(data)
    error("Undefined behavior to expand expression of type $expr_type with " *
          "measure data $data_type. If this functionality is needed consider " *
          "extending `_expand_measure`.")
    return
end

# Map temp point variable references to actual variables by adding a variable to
# the infinite model. This is used by expand(mref) to expand measures within an
# infinite model
function _add_mapped_point_variable(model::InfiniteModel,
                                    pvref::InfOptVariableRef,
                                    ivref::InfOptVariableRef, support::Tuple)
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
    expand(mref::InfOptVariableRef)::JuMP.AbstractJuMPScalar

Return a JuMP scalar function containing the explicit expansion of the measure
`mref`. This expansion is done according to the measure data. Note that
variables are added to the model as necessary to accomodate the expansion (i.e.,
point variables and reduced infinite variables are made as needed). Errors if
expansion is undefined for the measure data and/or the measure expression. If
desired this can be used in combination with [`measure`](@ref) to expand measures
on the fly.

This is useful for extensions that employ a custom optimizer_model since it
can be used evaluate measures before expressions are translated to the new model.
This method can also be extended to handle custom measure data types by extending
`InfiniteOpt._expand_measure` which should be of the form
`InfiniteOpt._expand_measure(::AbstractJuMPScalar, ::AbstractMeasureData, ::InfiniteModel, point_mapper::Function)`.
See the source code in InfiniteOpt/src/measures.jl for examples of how to do this.

**Example**
```julia
julia> tdata = DiscreteMeasureData(t, [0.5, 0.5], [0, 1])

julia> expr = expand(measure(g + z + T - h - 2, tdata))
0.5 g(0) + 0.5 g(1) + z + 0.5 T(0, x) + 0.5 T(1, x) - h(x) - 2
```
"""
function expand(mref::InfOptVariableRef)::JuMP.AbstractJuMPScalar
    return expand(mref, Val(variable_type(mref)))
end

function expand(mref::InfOptVariable, ::Val{MeasureRef})::JuMP.AbstractJuMPScalar
    return _expand_measure(measure_function(mref), measure_data(mref),
                           JuMP.owner_model(mref), _add_mapped_point_variable)
end

## Helper functions for expanding the measure references in expressions
# Measure reference
function _expand_measures(mref::InfOptVariableRef,
                          expand_model::JuMP.AbstractModel,
                          point_mapper::Function)::JuMP.AbstractJuMPScalar
    return _expand_measure(measure_function(mref), measure_data(mref),
                           expand_model, point_mapper)
end

# GenericAffExpr
function _expand_measures(expr::JuMP.GenericAffExpr{C, InfOptVariableRef},
                          expand_model::JuMP.AbstractModel,
                          point_mapper::Function)::JuMP.AbstractJuMPScalar where {C}
    # use a QuadExpr in case measures contain quadratic espressions
    quad = zero(JuMP.GenericQuadExpr{C, InfOptVariableRef})
    quad.aff.constant = expr.constant
    # add the variables to the expr, converting measures into expanded exprs
    for (var, coef) in expr.terms
        if isa(var, MeasureRef)
            JuMP.add_to_expression!(quad, coef, _expand_measure(measure_function(var),
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
function _expand_measures(expr::JuMP.GenericQuadExpr{C, InfOptVariableRef},
                          expand_model::JuMP.AbstractModel,
                          point_mapper::Function)::JuMP.GenericQuadExpr where {C}
    quad = zero(JuMP.GenericQuadExpr{C, InfOptVariableRef})
    quad.aff = _expand_measures(expr.aff, expand_model, point_mapper)
    # add the quadratic terms to the expr, converting measures into expanded exprs
    # note that this will error if the expanded forms are not linear or quadratic
    for (pair, coef) in expr.terms
        var_a = pair.a
        var_b = pair.b
        if variable_type(var_a) == MeasureRef
            var_a = _expand_measure(measure_function(var_a), measure_data(var_a),
                                    expand_model, point_mapper)
        end
        if variable_type(var_b) == MeasureRef
            var_b = _expand_measure(measure_function(var_b), measure_data(var_b),
                                    expand_model, point_mapper)
        end
        JuMP.add_to_expression!(quad, convert(JuMP.GenericQuadExpr{C,
                                InfOptVariableRef}, coef * var_a * var_b))
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

This is useful for extensions that employ a custom optimizer_model since it
can be used evaluate measures before `model` is translated into the new model.
This method can also be extended to handle custom measure data types by extending
`InfiniteOpt._expand_measure` which should be of the form
`InfiniteOpt._expand_measure(::AbstractJuMPScalar, ::AbstractMeasureData, ::InfiniteModel, point_mapper::Function)`.
See the source code in InfiniteOpt/src/measures.jl for examples of how to do this.

**Example**
```julia
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
        new_obj = _possible_convert(FiniteVariableRef, # change here
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
        new_func = _possible_convert(FiniteVariableRef, # change here
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
            bounds = model.constrs[cindex].bounds
            JuMP.delete(model, cref)
            JuMP.add_constraint(model, JuMP.build_constraint(error, new_func,
                                set, parameter_bounds = bounds), name)
        else
            JuMP.delete(model, cref)
            JuMP.add_constraint(model, JuMP.build_constraint(error, new_func,
                                set), name)
        end
        model.next_constr_index = curr_index
    end
    return
end
