import Redis from "../../../infrastructure/redis/redisDataProcessor";
import MLPRegressionModel from "../model/MLPRegressionModel";
import EvaluateResponse from "../../../models/response/evaluateResponse";
import fieldLabelFormatter from "../../queries/fieldLabelFormatter";
import redisDataProcessor from "../../../infrastructure/redis/redisDataProcessor";

const tf = require('@tensorflow/tfjs-node');

async function getEvaluateResponse(jobID: string, hubWeights: any): Promise<EvaluateResponse> {

    let redisKeys: any = await Redis.getRedisKey(jobID);
    redisKeys = await JSON.parse(redisKeys);
    const datasetRedisKey = redisKeys.datasetRedisKey;
    const optionsRedisKey = redisKeys.optionsRedisKey;
    const modelRedisKey = redisKeys.modelRedisKey;
    const weightsRedisKey = redisKeys.weightsRedisKey;

    const datasetStr = await Redis.getRedisKey(datasetRedisKey);
    const optionsStr = await Redis.getRedisKey(optionsRedisKey);
    const modelStr = await Redis.getRedisKey(modelRedisKey);
    const weights = hubWeights ? hubWeights : await Redis.getBuffer(weightsRedisKey);

    const options = await JSON.parse(optionsStr);
    var datasetJson = await JSON.parse(datasetStr);

    const width = 100
    const height = 100
    const depth = 1;
    const imageTensorArray = await fetchImages(datasetJson, width, height, depth);

    const flattenedLabelset = await
        datasetJson.ys
            .map((data: any) => {
                return Object.values(data)
            })
    const flattenedFeatureset = await
        datasetJson.xs
            .map((data: any) => {
                return Object.values(data)
            })

    var xDataset = await tf.data.array(flattenedFeatureset);
    var yDataset = await tf.data.array(flattenedLabelset);
    //xDataset.forEachAsync((e:any) => console.log(JSON.stringify(e)))

    if (imageTensorArray) {//multiInput model
        const image = await tf.data.array(imageTensorArray);
        xDataset = await tf.data.zip({ input_1: xDataset, input_2: image });
        yDataset = await tf.data.zip({ output: yDataset });
        var datasetObj = await tf.data.zip({ xs: xDataset, ys: yDataset })
    }
    else {//MLP model
        var datasetObj = await tf.data.zip({ xs: xDataset, ys: yDataset })
    }
    //await datasetObj.forEachAsync((e:any) => console.log(JSON.stringify(e)));

    const modelJson = await JSON.parse(modelStr);
    const EvaluateModel = await MLPRegressionModel.deserialize(modelJson, weights);
    const learningRate = options.optimizer.parameters.learning_rate;
    const optimizer = options.optimizer.name;
    const loss = options.compiler.parameters.loss;
    const metrics = options.compiler.parameters.metrics;
    await EvaluateModel.compile({ optimizer: tf.train[`${optimizer}`](learningRate), loss: loss, metrics: metrics });
    const datasetLength = Object.keys(datasetJson.xs).length;
    const evaluationSplit = options.optimizer.parameters.evaluation_split;
    const batchSize = options.optimizer.parameters.batch_size;
    const shuffle = options.optimizer.parameters.shuffle;
    const dataset = datasetObj.take(Math.floor(datasetLength * evaluationSplit)).shuffle(shuffle).batch(batchSize); //get evaluation dataset

    let result = await EvaluateModel.evaluateDataset(dataset);

    const responseMetrics = {
        loss: result[0].dataSync()[0],
        acc: result[1].dataSync()[0],
    }

    const evaluateResponse = {
        job: jobID,
        metrics: responseMetrics
    };
    return evaluateResponse
}

async function fetchImages(datasetJson: any, width: number, height: number, depth: number) {
    if (datasetJson.imageUIDlabel) {
        const label = fieldLabelFormatter.formatLabel(datasetJson.imageUIDlabel);
        const imgDataset = await Promise.all(datasetJson.xs.map(async (obj: any) => {
            const imageRedisKey = obj[label];
            const ubase64Image = await redisDataProcessor.getRedisKey(imageRedisKey);
            const imageBuffer = new Uint8Array(Buffer.from(ubase64Image, 'base64'));
            var result = await tf.node.decodeImage(imageBuffer);
            result = await tf.image.resizeNearestNeighbor(result, [width, height]);
            result = await ((tf.cast(result, 'float32').div(tf.scalar(255.0))));
            delete obj[label]
            return result;
        }));
        return imgDataset;
    }
    return
}

export default {
    getEvaluateResponse
}