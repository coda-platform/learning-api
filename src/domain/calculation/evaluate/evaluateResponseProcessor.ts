import Redis from "../../../infrastructure/redis/redisDataProcessor";
import UniModalRegressionModel from "../model/UniModalRegressionModel";
import EvaluateResponse from "../../../models/response/evaluateResponse";
import fieldLabelFormatter from "../../queries/fieldLabelFormatter";
import redisDataProcessor from "../../../infrastructure/redis/redisDataProcessor";
import MultiModalClassificationModel from "../model/MultiModalClassificationModel";

const tf = require('@tensorflow/tfjs-node');

async function getEvaluateResponse(jobID: string, hubWeights: any): Promise<EvaluateResponse> {
    const redisKeysStr = await Redis.getRedisKey(jobID);
    const redisKeys = JSON.parse(redisKeysStr);

    const [datasetStr, optionsStr, modelStr, weights] = await Promise.all([
        Redis.getRedisKey(redisKeys.datasetRedisKey),
        Redis.getRedisKey(redisKeys.optionsRedisKey),
        Redis.getRedisKey(redisKeys.modelRedisKey),
        hubWeights ? Promise.resolve(hubWeights) : Redis.getBuffer(redisKeys.weightsRedisKey)
    ]);

    const options = JSON.parse(optionsStr);
    const datasetJson = JSON.parse(datasetStr);

    const [width, height, depth] = options.transforms?.resizeImage ?
        [options.transforms.resizeImage.width,
        options.transforms.resizeImage.height,
        options.transforms.resizeImage.depth] : [100, 100, 1]

    const imageTensorArray = await fetchImages(datasetJson, width, height, depth);

    const flattenedLabelset = datasetJson.ys.map((data: any) => Object.values(data));
    const flattenedFeatureset = datasetJson.xs.map((data: any) => Object.values(data));

    const xDataset = tf.data.array(flattenedFeatureset);
    const yDataset = tf.data.array(flattenedLabelset);

    const modelJson = JSON.parse(modelStr);

    let datasetObj;
    let EvaluateModel;

    if (imageTensorArray) {
        const image = tf.data.array(imageTensorArray);
        const zippedXDataset = tf.data.zip({ input_1: xDataset, input_2: image });
        const zippedYDataset = tf.data.zip({ output: yDataset });
        datasetObj = tf.data.zip({ xs: zippedXDataset, ys: zippedYDataset });
        tf.dispose([image])
        EvaluateModel = await MultiModalClassificationModel.deserialize(modelJson, weights);
    } else {
        datasetObj = tf.data.zip({ xs: xDataset, ys: yDataset });
        EvaluateModel = await UniModalRegressionModel.deserialize(modelJson, weights);
    }

    const learningRate = options.optimizer.parameters.learning_rate;
    const optimizer = options.optimizer.name;
    const loss = options.compiler.parameters.loss;
    const metrics = options.compiler.parameters.metrics;

    EvaluateModel.compile({
        optimizer: tf.train[optimizer](learningRate),
        loss: loss,
        metrics: metrics
    });

    const datasetLength = Object.keys(datasetJson.xs).length;
    const evaluationSplit = options.optimizer.parameters.evaluation_split;
    const batchSize = options.optimizer.parameters.batch_size;
    const shuffle = options.optimizer.parameters.shuffle;
    const dataset = datasetObj.take(Math.floor(datasetLength * evaluationSplit)).shuffle(shuffle).batch(batchSize);

    const result = await EvaluateModel.evaluateDataset(dataset);

    const responseMetrics = {
        loss: result[0].dataSync()[0],
        acc: result[1].dataSync()[0]
    };

    tf.dispose([result, dataset, xDataset, yDataset, datasetObj,
        dataset, datasetObj, EvaluateModel, imageTensorArray]);

    return {
        job: jobID,
        metrics: responseMetrics
    };
}

async function fetchImages(datasetJson: any, width: number, height: number, depth: number) {
    if (!datasetJson.imageUIDlabel) return;

    const label = fieldLabelFormatter.formatLabel(datasetJson.imageUIDlabel);

    const imgDataset = await Promise.all(datasetJson.xs.map(async (obj: any) => {
        const imageRedisKey = obj[label];
        const ubase64Image = await redisDataProcessor.getRedisKey(imageRedisKey);
        const imageBuffer = new Uint8Array(Buffer.from(ubase64Image, 'base64'));

        let result = tf.node.decodeImage(imageBuffer);

        const resizedImage = tf.image.resizeNearestNeighbor(result, [width, height]);
        tf.dispose([result])

        const normalizedImage = resizedImage.cast('float32').div(tf.scalar(255.0));
        tf.dispose([resizedImage])

        delete obj[label];
        return normalizedImage;
    }));

    return imgDataset;
}

export default {
    getEvaluateResponse
}