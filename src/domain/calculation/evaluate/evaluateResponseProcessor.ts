import Redis from "../../../infrastructure/redis/redisDataProcessor";
import MLPRegressionModel from "../model/MLPRegressionModel";
import EvaluateResponse from "../../../models/response/evaluateResponse";
import fieldLabelFormatter from "../../queries/fieldLabelFormatter";
import redisDataProcessor from "../../../infrastructure/redis/redisDataProcessor";

const tf = require('@tensorflow/tfjs-node');

async function getEvaluateResponse(jobID: string, hubWeights: any): Promise<EvaluateResponse> {

    const redisKeysStr: any = await Redis.getRedisKey(jobID);
    const redisKeys = JSON.parse(redisKeysStr);

    const datasetStr = await Redis.getRedisKey(redisKeys.datasetRedisKey);
    const optionsStr = await Redis.getRedisKey(redisKeys.optionsRedisKey);
    const modelStr = await Redis.getRedisKey(redisKeys.modelRedisKey);
    const weights = hubWeights ? hubWeights : await Redis.getBuffer(redisKeys.weightsRedisKey);

    const options = JSON.parse(optionsStr);
    const datasetJson = JSON.parse(datasetStr);

    const width = 100;
    const height = 100;
    const depth = 1;
    const imageTensorArray = await fetchImages(datasetJson, width, height, depth);

    const flattenedLabelset = datasetJson.ys.map((data: any) => Object.values(data));
    const flattenedFeatureset = datasetJson.xs.map((data: any) => Object.values(data));

    let xDataset = tf.data.array(flattenedFeatureset);
    let yDataset = tf.data.array(flattenedLabelset);
    let datasetObj;

    if (imageTensorArray) { //multiInput model
        const image = tf.data.array(imageTensorArray);
        xDataset = tf.data.zip({ input_1: xDataset, input_2: image });
        yDataset = tf.data.zip({ output: yDataset });
        datasetObj = tf.data.zip({ xs: xDataset, ys: yDataset });
    } else { //MLP model
        datasetObj = tf.data.zip({ xs: xDataset, ys: yDataset });
    }

    const modelJson = JSON.parse(modelStr);
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

    const result = await EvaluateModel.evaluateDataset(dataset);

    // Ensure to dispose of the evaluation results after reading them
    const responseMetrics = {
        loss: result[0].dataSync()[0],
        acc: result[1].dataSync()[0],
    };

    tf.dispose(result);

    return {
        job: jobID,
        metrics: responseMetrics
    };
}

async function fetchImages(datasetJson: any, width: number, height: number, depth: number) {
    if (datasetJson.imageUIDlabel) {
        const label = fieldLabelFormatter.formatLabel(datasetJson.imageUIDlabel);
        const imgDataset = await Promise.all(datasetJson.xs.map(async (obj: any) => {
            const imageRedisKey = obj[label];
            const ubase64Image = await redisDataProcessor.getRedisKey(imageRedisKey);
            const imageBuffer = new Uint8Array(Buffer.from(ubase64Image, 'base64'));

            let decodedImage = tf.node.decodeImage(imageBuffer);
            const resizedImage = tf.image.resizeNearestNeighbor(decodedImage, [width, height]);
            const normalizedImage = tf.cast(resizedImage, 'float32').div(tf.scalar(255.0));

            // Dispose of intermediate tensors
            tf.dispose([decodedImage, resizedImage]);

            delete obj[label];
            return normalizedImage;
        }));

        return imgDataset;
    }
    return;
}

export default {
    getEvaluateResponse
}