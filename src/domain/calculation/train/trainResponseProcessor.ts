import redisDataProcessor from "../../../infrastructure/redis/redisDataProcessor";
import Redis from "../../../infrastructure/redis/redisDataProcessor";
import TrainResponse from "../../../models/response/trainResponse";
import fieldLabelFormatter from "../../queries/fieldLabelFormatter";
import MLPRegressionModel from "../model/MLPRegressionModel";

import * as tf from '@tensorflow/tfjs-node';

async function getTrainResponse(jobID: string, hubWeights: any): Promise<TrainResponse> {
    let redisKeys: any = await Redis.getRedisKey(jobID);
    redisKeys = await JSON.parse(redisKeys);

    const { datasetRedisKey, optionsRedisKey, modelRedisKey, weightsRedisKey } = redisKeys;

    const datasetStr = await Redis.getRedisKey(datasetRedisKey);
    const optionsStr = await Redis.getRedisKey(optionsRedisKey);
    const modelStr = await Redis.getRedisKey(modelRedisKey);
    const weights = hubWeights ? hubWeights : await Redis.getBuffer(weightsRedisKey);

    const options = await JSON.parse(optionsStr);
    const datasetJson = await JSON.parse(datasetStr);

    const [width, height, depth] = options.transforms?.resizeImage ?
        [options.transforms.resizeImage.width,
        options.transforms.resizeImage.height,
        options.transforms.resizeImage.depth] : [100, 100, 1]

    const imageTensorArray = await fetchImages(datasetJson, width, height, depth);

    const flattenedLabelset = datasetJson.ys.map((data: any) => Object.values(data));
    const flattenedFeatureset = datasetJson.xs.map((data: any) => Object.values(data));

    let xDataset = tf.data.array(flattenedFeatureset);
    let yDataset = tf.data.array(flattenedLabelset);
    let datasetObj;

    if (imageTensorArray) {
        const image = tf.data.array(imageTensorArray);
        xDataset = tf.data.zip({ input_1: xDataset, input_2: image });
        yDataset = tf.data.zip({ output: yDataset });
    }

    datasetObj = tf.data.zip({ xs: xDataset, ys: yDataset });

    const modelJson = await JSON.parse(modelStr);
    const TrainingModel = await MLPRegressionModel.deserialize(modelJson, weights);

    const { learning_rate, validation_split, evaluation_split, batch_size, epochs, shuffle } = options.optimizer.parameters;
    const optimizer = options.optimizer.name;
    const { loss, metrics } = options.compiler.parameters;

    await TrainingModel.compile({
        // @ts-ignore
        optimizer: tf.train[`${optimizer}`](learning_rate),
        loss: loss,
        metrics: metrics
    });

    const datasetLength = datasetJson.xs.length;
    const trainDatasetLength = Math.floor((1 - validation_split - evaluation_split) * datasetLength);
    const trainBatches = Math.floor(trainDatasetLength / batch_size);

    const dataset = datasetObj.skip(Math.floor(datasetLength * evaluation_split)).shuffle(shuffle).batch(batch_size);
    const trainDataset = dataset.take(trainBatches);
    const validationDataset = dataset.skip(trainBatches);

    const history = await TrainingModel.fitDataset(
        trainDataset,
        {
            epochs: epochs,
            validationData: validationDataset,
            verbose: 0
        }
    );

    const responseMetrics = {
        acc: history.history.acc[epochs - 1],
        loss: history.history.loss[epochs - 1],
        val_acc: history.history.val_acc[epochs - 1],
        val_loss: history.history.val_loss[epochs - 1],
    };

    const trainedWeights = await MLPRegressionModel.saveWeights(TrainingModel);

    // Explicitly dispose resources
    tf.dispose([xDataset, yDataset, datasetObj, dataset, trainDataset, validationDataset, TrainingModel, imageTensorArray]);

    return {
        job: jobID,
        weights: trainedWeights,
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

        let decodedImage = tf.node.decodeImage(imageBuffer);
        const resizedImage = tf.image.resizeNearestNeighbor(decodedImage, [width, height]);
        const normalizedImage = tf.cast(resizedImage, 'float32').div(tf.scalar(255.0));

        // Dispose intermediate tensors
        tf.dispose([decodedImage, resizedImage]);

        delete obj[label];
        return normalizedImage;
    }));

    return imgDataset;
}

export default {
    getTrainResponse,
}