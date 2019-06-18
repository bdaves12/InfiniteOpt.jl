# Extend Base.copy for new variable types
Base.copy(v::MeasureRef, new_model::InfiniteModel) = MeasureRef(new_model, v.index)

# Parse the string for displaying a measure
function _make_meas_name(meas::Measure)
    return string(meas.data.name, "(", JuMP.function_string(JuMP.REPLMode, meas.func), ")")
end

"""
    add_measure(model::InfiniteModel, v::Measure)
Add a measure to the `InfiniteModel` object in an analagous way to `JuMP.add_variable`.
"""
function add_measure(model::InfiniteModel, meas::Measure)
    model.next_meas_index += 1
    mref = MeasureRef(model, model.next_meas_index)
    model.measures[mref.index] = meas
    JuMP.name(mref, _make_meas_name(meas))
    return mref
end

# Parse the model pertaining to an expression
function _get_model_from_expr(expr::JuMP.AbstractJuMPScalar)
    if expr isa JuMP.AbstractVariableRef
        return expr.model
    elseif expr isa JuMP.GenericAffExpr
        aff_vars = [k for k in keys(expr.terms)]
        if length(aff_vars) > 0
            return aff_vars[1].model
        else
            return
        end
    elseif expr isa JuMP.GenericQuadExpr
        aff_vars = [k for k in keys(expr.aff.terms)]
        if length(aff_vars) > 0
            return aff_vars[1].model
        else
            var_pairs = [k for k in keys(expr.terms)]
            if length(var_pairs) > 0
                return var_pairs[1].a.model
            else
                return
            end
        end
    else
        return expr.m
    end
end

# TODO Make useful constructor functions
# Set a default weight function
# _w(t) = 1
#
# # Define constructor functions
# DiscreteMeasureData() = DiscreteMeasureData("measure", _w, [], [])
# DiscreteMeasureData(coeffs::Vector, supports::Array) = DiscreteMeasureData("measure", _w, coeffs, supports)
# DiscreteMeasureData(name::String, coeffs::Vector, supports::Array) = DiscreteMeasureData(name, _w, coeffs, supports)

"""
    measure(expr::Union{InfiniteExpr, MeasureRef}, data::AbstractMeasureData)
Implement a measure in an expression in a similar fashion to the `sum` method in JuMP.
"""
function measure(expr::Union{InfiniteExpr, MeasureRef}, data::AbstractMeasureData)
    meas = Measure(expr, data)
    model = _get_model_from_expr(expr)
    if model == nothing
        error("Expression contains no variables.")
    end
    return add_measure(model, meas)
end

"""
    JuMP.name(mref::MeasureRef)
Extend the `JuMP.name` function to accomodate measure references.
"""
JuMP.name(mref::MeasureRef) = mref.model.meas_to_name[mref.index]

# TODO Add manipulation functions like variables have --> not sure makes sense as they arne't used.
