package edu.dlsu.ccs.persondetection.train;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

import java.io.File;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.PointVector;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_ml;
import org.bytedeco.javacpp.opencv_ml.SVM;
import org.bytedeco.javacpp.opencv_objdetect.HOGDescriptor;

public class Driver {

	public static void main(String[] args) throws Exception {
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
//		HOGDescriptor hog = new HOGDescriptor(new Size(64, 128), new Size(8, 8),
//				new Size(8, 8), new Size(4, 4), 9);
		HOGDescriptor hog = new HOGDescriptor();
		for (String filename : positivesDirectory.list()) {
			FloatPointer descriptor = new FloatPointer();
			hog.compute(imread(positivesDirectory.getPath() + "\\" + filename),
					descriptor, new Size(8, 8), new Size(0, 0), new PointVector());
			Mat matDescriptor = new Mat(descriptor);
			Mat reshapedDescriptor = matDescriptor.reshape(0, 1);
			matDescriptor.close();
			trainData.push_back(reshapedDescriptor);
			responses.push_back(new Mat(new Size(1, 1), opencv_core.CV_32SC1, new Scalar(1.0)));
			
		}

		for (String filename : negativesDirectory.list()) {
			FloatPointer descriptor = new FloatPointer();
			hog.compute(imread(negativesDirectory.getPath() + "\\" + filename), descriptor, new Size(8, 8),
					new Size(0, 0), new PointVector());
			Mat matDescriptor = new Mat(descriptor);
			Mat reshapedDescriptor = matDescriptor.reshape(0, 1);
			matDescriptor.close();
			trainData.push_back(reshapedDescriptor);
			responses.push_back(new Mat(new Size(1, 1), opencv_core.CV_32SC1, new Scalar(-1.0)));
		}
		SVM svm = SVM.create();
		svm.setC(100);
		svm.setKernel(SVM.LINEAR);

		svm.train(trainData, opencv_ml.ROW_SAMPLE, responses);

		File testPositivesDirectory = new File("test/positives");
		File testNegativesDirectory = new File("test/negatives");

		System.out.println("testing");
		System.out.println("positives");
		for (String filename : testPositivesDirectory.list()) {
			FloatPointer descriptors = new FloatPointer();
			hog.compute(imread(testPositivesDirectory.getPath() + "\\" + filename), descriptors, new Size(8, 8),
					new Size(0, 0), new PointVector());
			Mat matDescriptor = new Mat(descriptors);
			Mat reshapedDescriptor = matDescriptor.reshape(0, 1);
			matDescriptor.close();
			System.out.println(svm.predict(reshapedDescriptor));
		}

		System.out.println("negatives");
		for (String filename : testNegativesDirectory.list()) {
			FloatPointer descriptors = new FloatPointer();
			hog.compute(imread(testNegativesDirectory.getPath() + "\\" + filename), descriptors, new Size(8, 8),
					new Size(0, 0), new PointVector());
			Mat matDescriptor = new Mat(descriptors);
			Mat reshapedDescriptor = matDescriptor.reshape(0, 1);
			matDescriptor.close();
			System.out.println(svm.predict(reshapedDescriptor));
		}
		hog.close();
		svm.save("svm_persondetect.xml");
		// HOGDescriptor humanDetector = new HOGDescriptor();
		// humanDetector.setSVMDetector(svm.getSupportVectors());
	}

}
