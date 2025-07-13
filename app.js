let model;

async function loadModel() {
  model = await tf.loadLayersModel('model/model.json');
  console.log("✅ Model loaded");
}

async function generateImage() {
  if (!model) return alert("Model not loaded!");

  const noise = tf.randomNormal([1, 100]); // 100-dim noise vector
  const prediction = model.predict(noise);

  const image = prediction.reshape([28, 28]);

  const canvas = document.getElementById('output');
  await tf.browser.toPixels(image, canvas);
}

loadModel();
