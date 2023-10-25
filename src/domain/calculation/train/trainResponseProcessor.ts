import redisDataProcessor from "../../../infrastructure/redis/redisDataProcessor";
import Redis from "../../../infrastructure/redis/redisDataProcessor";
import TrainResponse from "../../../models/response/trainResponse";
import fieldLabelFormatter from "../../queries/fieldLabelFormatter";
import MLPRegressionModel from "../model/MLPRegressionModel";

const tf = require('@tensorflow/tfjs-node');

async function getTrainResponse(jobID: string, hubWeights: any): Promise<TrainResponse> {

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

    const width = 512
    const height = 512
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
        xDataset = await tf.data.zip({ input1: xDataset, input2: image });
        yDataset = await tf.data.zip({ output: yDataset });
        var datasetObj = await tf.data.zip({ xs: xDataset, ys: yDataset })
    }
    else {//MLP model
        var datasetObj = await tf.data.zip({ xs: xDataset, ys: yDataset })
    }
    //await datasetObj.forEachAsync((e:any) => console.log(JSON.stringify(e)));

    const modelJson = await JSON.parse(modelStr);
    const TrainingModel = await MLPRegressionModel.deserialize(modelJson, weights);

    const learningRate = options.optimizer.parameters.learning_rate;
    const optimizer = options.optimizer.name;
    const loss = options.compiler.parameters.loss;
    const metrics = options.compiler.parameters.metrics;
    await TrainingModel.compile({ optimizer: tf.train[`${optimizer}`](learningRate), loss: loss, metrics: metrics });
    const datasetLength = datasetJson.xs.length;
    const validationSplit = options.optimizer.parameters.validation_split;
    const evaluationSplit = options.optimizer.parameters.evaluation_split;
    const batchSize = options.optimizer.parameters.batch_size;
    const epochs = options.optimizer.parameters.epochs;
    const shuffle = options.optimizer.parameters.shuffle;
    const dataset = datasetObj.skip(Math.floor(datasetLength * evaluationSplit)).shuffle(shuffle).batch(batchSize); //skip evaluation dataset
    const trainDatasetLength = Math.floor((1 - validationSplit - evaluationSplit) * datasetLength);
    const trainBatches = Math.floor(trainDatasetLength / batchSize);

    const trainDataset = dataset.take(trainBatches);
    const validationDataset = dataset.skip(trainBatches);

    let history = await TrainingModel.fitDataset(
        trainDataset,
        {
            epochs: epochs, validationData: validationDataset,
            // Disable progress bar which has some issues in TFJS
            verbose: 0
        }
    )


    const responseMetrics = {
        acc: history.history.acc[epochs - 1],
        loss: history.history.loss[epochs - 1],
        val_acc: history.history.val_acc[epochs - 1],
        val_loss: history.history.val_loss[epochs - 1],
    }
    //send weights as response
    const trainedWeights = await MLPRegressionModel.saveWeights(TrainingModel);

    const trainResponse = {
        job: jobID,
        weights: trainedWeights,
        metrics: responseMetrics
    };
    return trainResponse
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
    getTrainResponse,
}