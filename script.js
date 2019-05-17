async function fetchData() {
  const rawData = await fetch(
    'https://storage.googleapis.com/tfjs-tutorials/carsData.json'
  );
  const carData = await rawData.json();
  const cleaned = carData
    .map(car => ({
      mpg: car.Miles_per_Gallon,
      hp: car.Horsepower
    }))
    .filter(car => car.mpg != null && car.hp != null);

  return cleaned;
}

function createModel() {
  // Create sequential model
  const model = tf.sequential();

  // Single hidden layer
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

  // Middle hidden layer using non-linear sigmoid activation function
  model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }));

  // Output layer
  model.add(tf.layers.dense({ units: 1 }));

  return model;
}

function convertToTensor(data) {
  return tf.tidy(() => {
    // Shuffle
    tf.util.shuffle(data);

    // Convert to tensor
    const inputs = data.map(d => d.hp);
    const labels = data.map(d => d.mpg);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    // Normalize
    const inputMin = inputTensor.min();
    const inputMax = inputTensor.max();
    const labelMin = inputTensor.min();
    const labelMax = inputTensor.max();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMin,
      inputMax,
      labelMin,
      labelMax
    };
  });
}

async function trainModel(model, input, labels) {
  // Prep model for training
  model.compile({
    optimizer: tf.train.adam(), // algorithm to optimize values during training
    // optimizer: tf.train.adamax(),
    loss: tf.losses.meanSquaredError // tells model how well it's doing (i.e. feedback)
  });

  const batchSize = 64; // subsets of data that model sees each iteration
  const epochs = 50; // number of iterations

  return await model.fit(input, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}

function testModel(model, inputData, normalizationData) {
  const { inputMin, inputMax, labelMin, labelMax } = normalizationData;
  const [xs, ys] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100);
    const ys = model.predict(xs.reshape([100, 1]));

    // Unnormalize values
    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
    const unNormYs = ys.mul(labelMax.sub(labelMin)).add(labelMin);

    // dataSync() allows us to get values in tensors process them in JS
    return [unNormXs.dataSync(), unNormYs.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: ys[i] };
  });

  const originalPoints = inputData.map(d => ({
    x: d.hp,
    y: d.mpg
  }));

  tfvis.render.scatterplot(
    { name: 'Predicted & Original Values' },
    // {values: [predictedPoints, originalPoints], series: ['predicted', 'original']},
    {
      values: [originalPoints, predictedPoints],
      series: ['original', 'predicted']
    },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );
}

async function run() {
  // Load values
  const data = await fetchData();
  // Create data points
  const values = data.map(d => ({
    x: d.hp,
    y: d.mpg
  }));
  // Plot data
  tfvis.render.scatterplot(
    { name: 'Horsepower vs. MPG' },
    { values },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );

  // Create instance of model and visualize it
  const model = createModel();
  tfvis.show.modelSummary({ name: 'Model Summary' }, model);

  // Train model
  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;
  await trainModel(model, inputs, labels);
  console.log('yeet');

  // Test model
  testModel(model, data, tensorData);
}

run();
