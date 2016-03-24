package edu.dlsu.ccs.persondetection.train;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import org.opencv.objdetect.HOGDescriptor;

public class SVMTrainer {

	private HOGDescriptor hogDescriptor;
	private static final double POSITIVE_LABEL = 1.0;
	private static final double NEGATIVE_LABEL = 2.0;
	private SVM svm;

	public SVMTrainer() {
		hogDescriptor = new HOGDescriptor();
		hogDescriptor = new HOGDescriptor(new Size(80, 120), new Size(20, 20), new Size(10, 10), new Size(10, 10), 9);
		svm = SVM.create();
		svm.setC(0.016);
		// Mat classWeights = new Mat(1, 2, CvType.CV_32FC1);
		// classWeights.put(0, 0, 0.15);
		// classWeights.put(0, 1, 0.85);
		// svm.setClassWeights(classWeights);

		// svm.setTermCriteria(new TermCriteria(TermCriteria.EPS, 0, 1e-6));
		svm.setKernel(SVM.LINEAR);
	}

	public void start(String trainingPositivesDirectoryPath, String trainingNegativesDirectoryPath,
			String testingPositivesDirectoryPath, String testingNegativesDirectoryPath, String filename,
			boolean hardTrain, int maxIterations) throws Exception {
		String[] currentFalseNegatives = new String[0];
		String[] currentFalsePositives = new String[0];
		List<String> falseNegatives = new ArrayList<>();
		List<String> falsePositives = new ArrayList<>();
		int i = 0;
		// do{
		System.out.println("---Preparing training data---");
		Mat trainData = new Mat();
		Mat responses = new Mat();
		// if(currentFalseNegatives.length > 0 || currentFalsePositives.length >
		// 0){
		// System.out.println("Retraining");
		// }
		// else {
		// System.out.println("-----Training-----");
		// }

		trainFromDirectory(trainingPositivesDirectoryPath, trainingNegativesDirectoryPath, trainData, responses);

		if (falseNegatives.size() > 0) {
			// System.out.println("Preparing training data from false
			// negatives");
			// System.out.println("Generating descriptors for false negatives");
			// trainFromList(falseNegatives, trainData, responses, 1.0);
		}
		if (falsePositives.size() > 0) {
			// System.out.println("Preparing training data from false
			// positives");
			// System.out.println("Generating descriptors for false positives");
			// trainFromList(falsePositives, trainData, responses,
			// NEGATIVE_LABEL);
		}
		System.out.println("---Training---");
		svm.train(trainData, Ml.ROW_SAMPLE, responses);

		String[][] testResult = test(testingPositivesDirectoryPath, testingNegativesDirectoryPath);
		// currentFalseNegatives = testResult[0];
		currentFalsePositives = testResult[1];
		falseNegatives.addAll(Arrays.asList(currentFalseNegatives));
		falsePositives.addAll(Arrays.asList(currentFalsePositives));
		// } while(hardTrain && i++ < maxIterations &&
		// currentFalsePositives.length > 0);//(currentFalseNegatives.length > 0
		// || currentFalsePositives.length > 0));
		save(filename);
	}

	private void trainFromDirectory(String positivesDirectoryPath, String negativesDirectoryPath, Mat trainData,
			Mat responses) throws Exception {
		File positivesDirectory = new File(positivesDirectoryPath);
		File negativesDirectory = new File(negativesDirectoryPath);
		System.out.println("Preparing training data from directories");
		System.out.println("Generating descriptors for positive images");
		getDescriptorsForTrainingSet(positivesDirectory, trainData, responses, POSITIVE_LABEL);
		System.out.println("Generating descriptors for negative images");
		getDescriptorsForTrainingSet(negativesDirectory, trainData, responses, NEGATIVE_LABEL);
	}

	private void trainFromList(List<String> trainingDataPaths, Mat trainData, Mat responses, double classLabel)
			throws Exception {
		System.out.println(trainingDataPaths.size() + " training data from list");
		for (String filePath : trainingDataPaths) {
			Mat imageDescriptor = getDescriptorForImage(filePath);
			pushTrainDataAndResponse(imageDescriptor, trainData, responses, classLabel);
		}
	}

	private void getDescriptorsForTrainingSet(File trainingDataDirectory, Mat trainData, Mat responses,
			double classLabel) throws Exception {
		System.out.println(trainingDataDirectory.list().length + " training data from directory");
		for (String filename : trainingDataDirectory.list()) {
			File file = new File(trainingDataDirectory.getPath() + "\\" + filename);
			if (file.isDirectory()) {
				getDescriptorsForTrainingSet(file, trainData, responses, classLabel);
			} else {
				Mat imageDescriptor = getDescriptorForImage(trainingDataDirectory.getPath() + "\\" + filename);
				pushTrainDataAndResponse(imageDescriptor, trainData, responses, classLabel);
			}
		}
	}

	private void pushTrainDataAndResponse(Mat imageDescriptor, Mat trainData, Mat responses, double classLabel) {
		trainData.push_back(imageDescriptor);
		responses.push_back(new Mat(new Size(1, 1), CvType.CV_32SC1, new Scalar(classLabel)));
	}

	// get hog descriptor of gray-scale version of image specified by filePath
	private Mat getDescriptorForImage(String filePath) throws Exception {
		MatOfFloat imageDescriptor = new MatOfFloat();
		Mat image = Imgcodecs.imread(filePath);
		Mat grayScale = new Mat();
		Imgproc.cvtColor(image, grayScale, Imgproc.COLOR_BGR2GRAY);
		hogDescriptor.compute(grayScale, imageDescriptor);
		Mat reshapedImageDescriptor = imageDescriptor.reshape(0, 1);
		return reshapedImageDescriptor;
	}

	private String[][] test(String positivesDirectoryPath, String negativesDirectoryPath) throws Exception {
		System.out.println("---TESTING---");
		File testPositivesDirectory = new File(positivesDirectoryPath);
		File testNegativesDirectory = new File(negativesDirectoryPath);
		System.out.println("testing positive set");
		String[] falseNegatives = testDataSet(testPositivesDirectory, POSITIVE_LABEL);
		System.out.println(falseNegatives.length + " false negatives");
		System.out.println("testing negative set");
		String[] falsePositives = testDataSet(testNegativesDirectory, NEGATIVE_LABEL);
		System.out.println(falsePositives.length + " false positives");
		return new String[][] { falseNegatives, falsePositives };
	}

	private String[] testDataSet(File testDataDirectory, double expectedClassLabel) throws Exception {
		List<String> errors = new ArrayList<>();
		for (String filename : testDataDirectory.list()) {
			String filePath = testDataDirectory.getPath() + "\\" + filename;

			File file = new File(filePath);
			if (file.isDirectory()) {
				errors.addAll(Arrays.asList(testDataSet(file, expectedClassLabel)));
			} else {
				Mat imageDescriptor = getDescriptorForImage(filePath);
				float actualClassLabel = svm.predict(imageDescriptor);
				if (actualClassLabel != (float) expectedClassLabel) {
					// System.out.println("Error on " + filename + "; " +
					// "expected: " + expectedClassLabel + " actual: "
					// + actualClassLabel);
					// System.out.println("Adding to re-training set");
					errors.add(filePath);
				}
			}
		}
		return errors.toArray(new String[errors.size()]);
	}

	private void save(String filename) {
		svm.save("svm_persondetect.xml");
	}
}
