export type variableType = 'categorical' | 'continuous' | 'tensor'
const tf = require('@tensorflow/tfjs-node');
/**
* Builds and returns Multi Layer Perceptron Regression Model.
*
* @param {number} mlpInputShape The input shape of the mlp model.
* @returns {tf.Sequential} The multi layer perceptron regression mode  l.
*/
async function createMultiInputClassificationModel(mlpInputShape: [number, number, number, number]) {

  const [length, width, height, depth] = mlpInputShape

  // input1
  const myInput1 = tf.input({ shape: length, name: 'input_1' });
  const myInput1Dense1 = tf.layers.dense({ units: 8, activation: 'relu', name: 'dense_1' }).apply(myInput1);
  const myInput1Dense2 = tf.layers.dense({ units: 8, activation: 'relu', name: 'dense_2' }).apply(myInput1Dense1);
  const output1 = tf.layers.dense({ units: 4, activation: 'relu', name: 'output_1' }).apply(myInput1Dense2)

  // input2
  const myInput2 = tf.input({ shape: [width, height, depth], name: 'input_2' });
  const conv2d_1 = tf.layers.conv2d({ filters: 16, kernelSize: 3, activation: 'relu', name: 'conv2d_1' }).apply(myInput2);
  const conv2d_1_bn = tf.layers.batchNormalization({ name: 'batch_norm_1' }).apply(conv2d_1);
  const conv2d_1_bn_pooled = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(conv2d_1_bn);

  const flatten = tf.layers.flatten({ name: 'flatten' }).apply(conv2d_1_bn_pooled);
  const output2 = tf.layers.dense({ units: 4, activation: 'relu', name: 'output_2' }).apply(flatten)

  const concat = tf.layers.concatenate({ name: 'concat' }).apply([output1, output2]);
  const concat_dense = tf.layers.dense({ units: 16, activation: 'relu', name: 'concat_dense' }).apply(concat)
  // name last layer 'output' 
  const output = tf.layers.dense({ units: 1, activation: 'sigmoid', name: 'output' }).apply(concat_dense)
  var model = tf.model({ inputs: [myInput1, myInput2], outputs: output });

  await model.compile({ optimizer: tf.train.adam(10e-5), loss: 'binaryCrossentropy' });
  return model
}

async function saveWeights(model: any) {
  let result = await model.save(tf.io.withSaveHandler(async (modelArtifacts: any) => modelArtifacts));
  result.weightData = Buffer.from(result.weightData);
  return result.weightData;
}

async function serialize(model: any) {
  let result = await model.save(tf.io.withSaveHandler(async (modelArtifacts: any) => modelArtifacts));
  delete result.weightData;
  return result;
}

async function deserialize(json: any, weights: any) {
  const weightData = new Uint8Array(Buffer.from(weights)).buffer;
  const modelArtifacts = {
    modelTopology: json.modelTopology,
    weightSpecs: json.weightSpecs,
    weightData: weightData
  }
  let loadedModel = await tf.loadLayersModel(tf.io.fromMemory(modelArtifacts));
  return loadedModel;
}

export default {
  createMultiInputClassificationModel, serialize, deserialize, saveWeights
}