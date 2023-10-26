import dataResponseProcessor from "../domain/calculation/prepare/dataReponseProcessor";
import FieldInfo from "../models/fieldInfo";
import Filter from "../models/request/filter";
import Field from "../models/request/field";
import PrepareRequestBody from "../models/request/prepareRequestBody";
import PrepareResponse from "../models/response/prepareResponse";
import queryDataRepository from "../repositories/data/queryDataRepository";
import fieldsRepository from "../repositories/fields/fieldsRepository";
import filterFieldsRepository from "../repositories/fields/filterFieldsRepository";
import TrainRequestBody from "../models/request/trainRequestBody";
import trainResponseProcessor from "../domain/calculation/train/trainResponseProcessor";
import EvaluateRequestBody from "../models/request/evaluateRequestBody";
import evaluateResponseProcessor from "../domain/calculation/evaluate/evaluateResponseProcessor"

async function getFieldTypeAndErrors(prepareRequest: PrepareRequestBody, filterFieldsNoErrors: Map<Filter, FieldInfo>) {

    const fields = await fieldsRepository.getFieldsDataFromRequest(prepareRequest, filterFieldsNoErrors);

    const fieldRequestErrors = new Array<PrepareResponse>();
    fields.forEach((value, key) => {
        if (value instanceof Error) {
            const errorFormatted: PrepareResponse = { job: '', error: value, query: key.path };
            fieldRequestErrors.push(errorFormatted);
        }
    });
    return { fieldRequestErrors, fields };
}

async function getFilterTypesAndErrors(prepareRequest: PrepareRequestBody) {
    const filterFields = await filterFieldsRepository.getFieldsDataFromRequest(prepareRequest);

    const filterRequestErrors = new Array<PrepareResponse>();
    filterFields.forEach((value, key) => {
        if (value instanceof Error) {
            const errorFormatted: PrepareResponse = { job: '', error: value, query: key.path };
            filterRequestErrors.push(errorFormatted);
        }
    });
    return { filterRequestErrors, filterFields };
}

async function getPrepare(prepareRequest: PrepareRequestBody) {
    const { filterRequestErrors, filterFields } = await getFilterTypesAndErrors(prepareRequest);

    if (filterRequestErrors.length > 0) return filterRequestErrors;
    const filterFieldsNoErrors = new Map<Filter, FieldInfo>(filterFields as any);

    const { fieldRequestErrors, fields } = await getFieldTypeAndErrors(prepareRequest, filterFieldsNoErrors);

    if (fieldRequestErrors.length > 0) return fieldRequestErrors;
    const fieldsNoErrors = new Map<Field, FieldInfo>(fields as any);

    const queryDataResults = await queryDataRepository.executeQueries(prepareRequest, fieldsNoErrors, filterFieldsNoErrors);
    const response = await dataResponseProcessor.getPrepareResponses(prepareRequest, queryDataResults, fieldsNoErrors);

    return response;
}

async function getTrain(TrainRequest: TrainRequestBody) {
    const job = TrainRequest.job;
    const weights = TrainRequest.weights;
    const response = trainResponseProcessor.getTrainResponse(job, weights);
    return response;
}

async function getEvaluate(evaluateRequest: EvaluateRequestBody) {
    const job = evaluateRequest.job;
    const weights = evaluateRequest.weights;
    const response = evaluateResponseProcessor.getEvaluateResponse(job, weights);
    return response;
}

export default {
    getPrepare, getTrain, getEvaluate
}