const tf = require('@tensorflow/tfjs-node')
const model = tf.sequential();
model.add(tf.layers.dense({units:1,inputShape:[1]}));
model.compile({loss:tf.losses.meanSquaredError,optimizer:tf.train.sgd(0.1)})
const xs = [1,2,3,4]
const ys = [1,3,5,7]
model.fit(tf.tensor(xs),tf.tensor(ys),{
    batchSize: 2,
    epochs: 100
}).then(res=>{
    console.log(res)
    model.predict(tf.tensor([6,7,8,9])).print()
})
