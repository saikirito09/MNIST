import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

// Set up the Spark session
val spark = SparkSession.builder().appName("MNIST Classifier").master("local[*]").getOrCreate()

// Load the MNIST dataset
val trainData = spark.read.format("csv").option("header", "true").load("train.csv")
val testData = spark.read.format("csv").option("header", "true").load("test.csv")

// Convert the data types of the features and labels
val trainFeatures = trainData.columns.tail.map(c => trainData(c).cast("double"))
val trainLabels = trainData("label").cast("double")
val testFeatures = testData.columns.tail.map(c => testData(c).cast("double"))
val testLabels = testData("label").cast("double")

// Convert the training and testing labels to a one-hot encoding
val trainLabelsOH = trainLabels.rdd.map(label => {
  val arr = Array.ofDim[Double](10)
  arr(label.toInt) = 1.0
  arr
}).toDF("label")
val testLabelsOH = testLabels.rdd.map(label => {
  val arr = Array.ofDim[Double](10)
  arr(label.toInt) = 1.0
  arr
}).toDF("label")

// Define the neural network architecture
val layers = Array[Int](784, 512, 256, 10)

// Set up the pipeline
val assembler = new VectorAssembler().setInputCols(trainFeatures.map(_.toString)).setOutputCol("features")
val classifier = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(Array("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"))
val pipeline = new Pipeline().setStages(Array(assembler, classifier, labelConverter))

// Train the model
val model = pipeline.fit(trainData)

// Make predictions on the testing data
val predictions = model.transform(testData)

// Evaluate the model
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test set accuracy = ${accuracy}")
