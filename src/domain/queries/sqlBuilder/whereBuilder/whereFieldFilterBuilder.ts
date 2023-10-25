import FieldInfo from "../../../../models/fieldInfo";
import Field from "../../../../models/request/field";
import Filter from "../../../../models/request/filter";
import Selector from "../../../../models/request/selector";
import possibleComputedFilters from "../../../possibleComputedFilters";
import jsonFieldValuePathCompiler from "../../../queries/jsonFieldValuePathCompiler";
import jsonFilterOperatorFormatter from "../../../queries/jsonFilterOperatorFormatter";
import jsonQueryValueFormatter from "../../../queries/jsonQueryValueFormatter";
import arrayFieldDetector from "../../fields/arrayFieldDetector";
import WhereJsonArrayFormatterBuilder from "./whereJsonArrayFormatterBuilder";
import indexArrayFieldDetector from "../../fields/indexArrayFieldDetector"

function getFilterNormalized(filter: Filter, filterFields: Map<Filter, FieldInfo>, selector: Selector): string {
    const fieldInfo = filterFields.get(filter);
    if (!fieldInfo) throw new Error('No matching field for filter.')
    const isArrayField = arrayFieldDetector.isArrayField(filter.path);
    const isIndexArrayField = indexArrayFieldDetector.isIndexArrayField(filter.path);
    const jsonFieldValuePathCompiled = isArrayField
        ? new WhereJsonArrayFormatterBuilder(filter.path, selector.label).build()
        : jsonFieldValuePathCompiler.getPathCompiled(filter.path);

    const filterValue = isIndexArrayField
        ? jsonQueryValueFormatter.formatIndexValueForSql(filter.value, filter.path)
        : jsonQueryValueFormatter.formatValueForSql(filter, fieldInfo);

    try {
        const sqlOperand = isIndexArrayField
            ? "@>"
            : jsonFilterOperatorFormatter.formatOperatorForSql(filter);
        const cast = fieldInfo.type
        return isIndexArrayField
            ? `${jsonFieldValuePathCompiled} ${sqlOperand} ${filterValue}`
            : `(${jsonFieldValuePathCompiled})::${cast} ${sqlOperand} ${filterValue}`;
    } catch (err) {
        const error = err as Error
        console.error(`Error while building filter ${filter.path}: ${error.message}`);
        return "1=1"
    }

}

function build(selector: Selector, filterFieldTypes: Map<Filter, FieldInfo>, possibleComputedField?: Field) {
    const filtersNormalized = selector.filters.map(f => getFilterNormalized(f, filterFieldTypes, selector));

    if (possibleComputedField) {
        const addedFieldFilter = possibleComputedFilters.possibleComputedFilters.get(possibleComputedField.path);

        if (addedFieldFilter) {
            filtersNormalized.push(addedFieldFilter);
        }
    }

    return filtersNormalized.join(' AND ');
}

export default {
    build
}