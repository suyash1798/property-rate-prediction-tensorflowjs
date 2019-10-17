require('@tensorflow/tfjs-node');

const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');


function Knn(features, labels, predictionPoint, k) {

    const {mean, variance} = tf.moments(features, 0);

    const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5))

    return features
        .sub(mean)
        .div(variance.pow(0.5))
        .sub(scaledPrediction)
        .pow(2)
        .sum(1)
        .pow(0.5)
        .expandDims(1)
        .concat(labels, 1)
        .unstack()
        .sort((a, b) => a.arraySync()[0] > b.arraySync()[0] ? 1 : -1)
        .slice(0, k)
        .reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k;
}

let {features, labels, testFeatures, testLabels} = loadCSV('kc_house_data.csv', {
    shuffle: true,
    splitTest: 10000,
    dataColumns: ['date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15'],
    labelColumns: ['price']
});

features = tf.tensor(features);
labels = tf.tensor(labels);

features.print();
labels.print();
tf.tensor(testFeatures[0]).print();

// var result =
// var i = 0;
// testFeatures.forEach((testpoint) => {
var result = Knn(features, labels, tf.tensor(testFeatures[0]), 10000);
// console.log(result);
const err = (testLabels[0][0] - result) / testLabels[0][0];
// console.log('Guess', testLabels, testFeatures);
console.log('Error', err, result, testLabels[0][0]);
// i++;
// });
