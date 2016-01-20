package edu.dlsu.ccs.persondetection.train;

import java.io.File;
import java.io.IOException;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.PointVector;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_ml;
import org.bytedeco.javacpp.opencv_ml.SVM;
import org.bytedeco.javacpp.opencv_objdetect.HOGDescriptor;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;

public class Driver {

	public static void main(String[] args) throws IOException {
		// System.loadLibrary(opencv_core.Core.NATIVE_LIBRARY_NAME);

		File positivesDirectory = new File("train/positives");
		File negativesDirectory = new File("train/negatives");
		if (!positivesDirectory.isDirectory()) {
			throw new RuntimeException("Directory for positive images is missing");
		}
		if (!negativesDirectory.isDirectory()) {
			throw new RuntimeException("Directory for negative images is missing");
		}

		Mat trainData = new Mat();
		Mat responses = new Mat();
		HOGDescriptor hog = new HOGDescriptor();

		for (String filename : positivesDirectory.list()) {
			FloatPointer descriptor = new FloatPointer();
			hog.compute(imread(positivesDirectory.getPath() + "\\" + filename),
					descriptor, new Size(8, 8), new Size(0, 0), new PointVector());
			trainData.push_back(new Mat(descriptor).reshape(0, 1));
			responses.push_back(Mat.ones(new Size(1, 1), opencv_core.CV_32SC1).asMat());
			
		}

		for (String filename : negativesDirectory.list()) {
			FloatPointer descriptor = new FloatPointer();
			hog.compute(imread(negativesDirectory.getPath() + "\\" + filename), descriptor, new Size(8, 8),
					new Size(0, 0), new PointVector());
			trainData.push_back(new Mat(descriptor).reshape(0, 1));
			responses.push_back(Mat.zeros(new Size(1, 1), opencv_core.CV_32SC1).asMat());
		}
		SVM svm = SVM.create();
		svm.train(trainData, opencv_ml.ROW_SAMPLE, responses);

		File testPositivesDirectory = new File("test/positives");
		File testNegativesDirectory = new File("test/negatives");

		System.out.println("testing");
		System.out.println("positives");
		for (String filename : testPositivesDirectory.list()) {
			FloatPointer descriptors = new FloatPointer();
			hog.compute(imread(testPositivesDirectory.getPath() + "\\" + filename), descriptors, new Size(8, 8),
					new Size(0, 0), new PointVector());
			System.out.println(svm.predict(new Mat(descriptors).reshape(0, 1)));
		}

		System.out.println("negatives");
		for (String filename : testNegativesDirectory.list()) {
			FloatPointer descriptors = new FloatPointer();
			hog.compute(imread(testNegativesDirectory.getPath() + "\\" + filename), descriptors, new Size(8, 8),
					new Size(0, 0), new PointVector());
			System.out.println(svm.predict(new Mat(descriptors).reshape(0, 1)));
		}

		svm.save("svm_persondetect.xml");
		// HOGDescriptor humanDetector = new HOGDescriptor();
		// humanDetector.setSVMDetector(svm.getSupportVectors());
	}

}
