const path = require('path')
const fs = require('fs')
//const fr = require('face-recognition')
const {
  fr,
  getAppdataPath,
  ensureAppdataDirExists
} = require('./commonsForFaceRecognition')

fr.winKillProcessOnExit()
ensureAppdataDirExists()

const numTrainingFaces = 10
const trainedModelFile = `faceRecognition1Model_t${numTrainingFaces}_150.json`
const trainedModelFilePath = path.resolve(getAppdataPath(), trainedModelFile)

const dataPath = path.resolve('./data/faces')
const classNames = ['hung', 'howard', 'lennard', 'raj', 'sheldon', 'thu']

const allFiles = fs.readdirSync(dataPath)
const imagesByClass = classNames.map(c =>
  allFiles
    .filter(f => f.includes(c))
    .map(f => path.join(dataPath, f))
    .map(fp => fr.loadImage(fp))
)
/*
const detector = fr.FaceDetector()
const targetSize = 150
const faceImages = detector.detectFaces(image, targetSize)
faceImages.forEach((img, i) => fr.saveImage(`face_${i}.png`, img))
*/

const trainDataByClass = imagesByClass.map(imgs => imgs.slice(0, numTrainingFaces))
const testDataByClass = imagesByClass.map(imgs => imgs.slice(numTrainingFaces))

const recognizer = fr.FaceRecognizer()

if (!fs.existsSync(trainedModelFilePath)) {
  console.log('%s not found, start training recognizer...', trainedModelFile)

  trainDataByClass.forEach((faces, label) => {
    const name = classNames[label]
    recognizer.addFaces(faces, name)
  })

  fs.writeFileSync(trainedModelFilePath, JSON.stringify(recognizer.serialize()));
} else {
  console.log('found %s, loading model', trainedModelFile)

  recognizer.load(require(trainedModelFilePath))

  console.log('imported the following descriptors:')
  console.log(recognizer.getDescriptorState())
}

/*
trainDataByClass.forEach((faces, label) => {
  const name = classNames[label]
  recognizer.addFaces(faces, name)
})

/*Save
const modelState = recognizer.serialize()
fs.writeFileSync('model.json', JSON.stringify(modelState))

//LOAD

const modelState = require('./model.json')
recognizer.load(modelState)
//*/

const errors = classNames.map(_ => [])
testDataByClass.forEach((faces, label) => {
  const name = classNames[label]
  console.log()
  console.log('testing %s', name)
  faces.forEach((face, i) => {
    const prediction = recognizer.predictBest(face)
    console.log('%s (%s)', prediction.className, prediction.distance)

    // count number of wrong classifications
    if (prediction.className !== name) {
      errors[label] = errors[label] + 1
    }
  })
})

// print the result
const result = classNames.map((className, label) => {
  const numTestFaces = testDataByClass[label].length
  const numCorrect = numTestFaces - errors[label].length
  const accuracy = parseInt((numCorrect / numTestFaces) * 10000) / 100
  return `${className} ( ${accuracy}% ) : ${numCorrect} of ${numTestFaces} faces have been recognized correctly`
})
console.log('result:')
console.log(result)