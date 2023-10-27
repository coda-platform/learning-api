const tf = require('@tensorflow/tfjs-node');

// Builds and returns a Multi Layer Perceptron Model with tabular inputs.
async function createUniModalRegressionModel(inputShape: number[]) {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: inputShape,
    activation: 'relu',
    units: 20,
  }));
  model.add(tf.layers.dropout({
    rate: 0.1,
  }))
  model.add(tf.layers.dense({
    activation: 'relu',
    units: 10,
  }));
  model.add(tf.layers.dense({
    activation: 'sigmoid',
    units: 1,
  }));
  model.compile({ optimizer: tf.train.adam(0.0005), loss: 'meanSquaredError' });
  return model;
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
  if (!Buffer.isBuffer(weights)) {
    var weightData = new Uint8Array(Buffer.from(weights)).buffer;
  }
  else {
    var weightData = new Uint8Array(weights).buffer;
  }
  const modelArtifacts = {
    modelTopology: json.modelTopology,
    weightSpecs: json.weightSpecs,
    weightData: weightData
  }

  const loadedModel = await tf.loadLayersModel(tf.io.fromMemory(modelArtifacts));

  return loadedModel;
}

export default {
  createUniModalRegressionModel, serialize, deserialize, saveWeights
}