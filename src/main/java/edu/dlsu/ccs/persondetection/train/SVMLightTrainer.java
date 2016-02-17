package edu.dlsu.ccs.persondetection.train;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;

import jnisvmlight.KernelParam;
import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.LearnParam;
import jnisvmlight.SVMLightInterface;
import jnisvmlight.SVMLightModel;
import jnisvmlight.TrainingParameters;

public class SVMLightTrainer {

	private static final double POSITIVE_LABEL = 1.0;
	private static final double NEGATIVE_LABEL = -1.0;
	private HOGDescriptor hogDescriptor;

	public SVMLightTrainer() {
		hogDescriptor = new HOGDescriptor();
	}

	public void start(String trainingPositivesDirectoryPath, String trainingNegativesDirectoryPath,
			String testingPositivesDirectoryPath, String testingNegativesDirectoryPath, String filename,
			boolean hardTrain, int maxIterations) throws Exception {
		String[] currentFalseNegatives = new String[0];
		String[] currentFalsePositives = new String[0];
		List<String> falseNegatives = new ArrayList<>();
		List<String> falsePositives = new ArrayList<>();
		System.out.println(hardTrain);
		int i = 0;
		// do{
		Mat trainData = new Mat();
		Mat responses = new Mat();
		if (currentFalseNegatives.length > 0 || currentFalsePositives.length > 0) {
			System.out.println("Retraining");
		} else {
			System.out.println("-----Training-----");
		}

		List<LabeledFeatureVector> featureVectorsFromDirectories = trainFromDirectory(trainingPositivesDirectoryPath,
				trainingNegativesDirectoryPath, trainData, responses);

		train(featureVectorsFromDirectories.toArray(new LabeledFeatureVector[featureVectorsFromDirectories.size()]));
		System.out.println("Training from list");
		// System.out.println("Training false negatives");
		// trainFromList(falseNegatives, trainData, responses, 1.0);
		System.out.println("Training false positives");
		// List<LabeledFeatureVector> featureVectorsFromList =
		// trainFromList(falsePositives, trainData, responses, -1.0);

		// new SVMLightTrainer().train(trainData, responses);
		// svm.train(trainData, Ml.ROW_SAMPLE, responses);

		// String[][] testResult = test(testingPositivesDirectoryPath,
		// testingNegativesDirectoryPath);
		// currentFalseNegatives = testResult[0];
		// currentFalsePositives = testResult[1];
		// falseNegatives.addAll(Arrays.asList(currentFalseNegatives));
		// falsePositives.addAll(Arrays.asList(currentFalsePositives));
		// } while(hardTrain && i++ <
		// maxIterations);//(currentFalseNegatives.length > 0 ||
		// currentFalsePositives.length > 0));
	}

	private List<LabeledFeatureVector> trainFromDirectory(String positivesDirectoryPath, String negativesDirectoryPath,
			Mat trainData, Mat responses) throws Exception {
		File positivesDirectory = new File(positivesDirectoryPath);
		File negativesDirectory = new File(negativesDirectoryPath);
		System.out.println("Processing data from directories");
		System.out.println("Generating descriptors for positive images");
		List<LabeledFeatureVector> positiveFeatureVectors = getFeatureVectorsForTrainingSet(positivesDirectory,
				trainData, responses, POSITIVE_LABEL);
		System.out.println("Generating descriptors for negative images");
		List<LabeledFeatureVector> negativeFeatureVectors = getFeatureVectorsForTrainingSet(negativesDirectory,
				trainData, responses, NEGATIVE_LABEL);
		positiveFeatureVectors.addAll(negativeFeatureVectors);
		return positiveFeatureVectors;
	}

	private LabeledFeatureVector[] trainFromList(List<String> trainingDataPaths, Mat trainData, Mat responses,
			double classLabel) throws Exception {
		System.out.println(trainingDataPaths.size() + " training data from list");
		List<LabeledFeatureVector> featureVectors = new ArrayList<>();
		for (String filePath : trainingDataPaths) {
			double[] imageDescriptor = getDescriptorForImage(filePath);
			featureVectors.add(
					new LabeledFeatureVector(classLabel, new int[] { 1, imageDescriptor.length }, imageDescriptor));
		}
		return featureVectors.toArray(new LabeledFeatureVector[featureVectors.size()]);
	}

	private List<LabeledFeatureVector> getFeatureVectorsForTrainingSet(File trainingDataDirectory, Mat trainData,
			Mat responses, double classLabel) throws Exception {
		System.out.println(trainingDataDirectory.list().length + " training data from directory");
		List<LabeledFeatureVector> featureVectors = new ArrayList<>();
		for (String filename : trainingDataDirectory.list()) {
			double[] imageDescriptor = getDescriptorForImage(trainingDataDirectory.getPath() + "\\" + filename);
			int[] dims = new int[imageDescriptor.length];
			Arrays.fill(dims, 1);
			featureVectors.add(
					new LabeledFeatureVector(classLabel, dims, imageDescriptor));
		}
		return featureVectors;
	}

	// get hog descriptor of gray-scale version of image specified by filePath
	private double[] getDescriptorForImage(String filePath) throws Exception {
		MatOfFloat imageDescriptor = new MatOfFloat();
		Mat image = Imgcodecs.imread(filePath);
		Mat grayScale = new Mat();
		Imgproc.cvtColor(image, grayScale, Imgproc.COLOR_BGR2GRAY);
		hogDescriptor.compute(grayScale, imageDescriptor);

		float[] floatValues = imageDescriptor.toArray();
		double[] doubleValues = new double[floatValues.length];
		for (int i = 0; i < floatValues.length; i++) {
			doubleValues[i] = (double) floatValues[i];
		}
		return doubleValues;
	}

	public void train(LabeledFeatureVector[] trainData) {
		SVMLightInterface trainer = new SVMLightInterface();

		
		KernelParam kernelParameters = new KernelParam();
		kernelParameters.kernel_type = KernelParam.LINEAR;
		
		LearnParam learningParameters = new LearnParam();
		learningParameters.epsilon_crit = 100; 
		learningParameters.type = LearnParam.CLASSIFICATION;
		
		TrainingParameters svmParameters = new TrainingParameters();
		svmParameters.setKernelParameters(kernelParameters);
		svmParameters.setLearningParameters(learningParameters);
		
		System.out.println("Training svm");
		SVMLightModel svm = trainer.trainModel(trainData, svmParameters);
		System.out.println("Saving SVM");
		svm.writeModelToFile("svm_persondetect.dat");
	}
}
