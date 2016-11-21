library('h2o')
h2o.init(nthreads=-1)
# import data
negative_data_set 		<- read.table('../images_as_txt/negativeBw.txt')
positive_data_set 		<- read.table('../images_as_txt/positiveBw.txt')
dataSet 		 		<- rbind(negative_data_set,positive_data_set)
# shuffle
dataSet 				<- dataSet[sample(nrow(dataSet)),]
trainingData			<- dataSet[1:floor(nrow(dataSet)*0.7),]
testingData				<- dataSet[floor(nrow(dataSet)*0.7):nrow(dataSet),]
trainingInput           <- trainingData[1:ncol(trainingData)-1]
trainingOutput          <- trainingData[ncol(trainingData)]
# create neural network
inputTypes      		<- names(trainingInput)
outputTypes             <- names(trainingOutput)
trainingData            <- as.h2o(trainingData)
testingData 			<- as.h2o(testingData)
neuralNetwork 			<- h2o.deeplearning(x=inputTypes, y=outputTypes, training_frame=trainingData, epochs=25)
# test network
testingOutput			<- h2o.predict(neuralNetwork, testingData)
ME						<- sum(abs(testingOutput-testingData[outputTypes]))/nrow(testingOutput)
MSE						<- sum((testingOutput-testingData[outputTypes])^2)/nrow(testingOutput)
adjustedOutput			<- testingOutput > 0.5
percentAccuracy			<- sum(adjustedOutput==testingData[outputTypes])/nrow(testingOutput)*100
# get some measurements
truePositives			<- adjustedOutput && testingData[outputTypes]
falsePositives			<- adjustedOutput && !testingData[outputTypes]
trueNegatives			<- !adjustedOutput && !testingData[outputTypes]
falseNegatives 			<- !adjustedOutput && testingData[outputTypes]

truePositivesCount 		<- sum(truePositives)
falsePositivesCount		<- sum(falsePositives)
trueNegativesCount		<- sum(trueNegatives)
falseNegativesCount 	<- sum(falseNegatives)
total					<- nrow(adjustedOutput)

sensitivity				<- truePositivesCount/(truePositivesCount+falseNegativesCount)
specificity				<- trueNegativesCount/(trueNegativesCount+falsePositivesCount)
accuracy 				<- (truePositivesCount+trueNegativesCount)/total
precision				<- truePositivesCount/(truePositivesCount+falsePositivesCount)


# h2o.shutdown()