package edu.dlsu.ccs.persondetection.train;

import org.opencv.core.Core;

public class Driver {

	public static void main(String[] args) throws Exception {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		SVMTrainer trainer = new SVMTrainer();
		trainer.start("data/training/positives", 
						"data/training/negatives", 
						"data/testing/positives", 
						"data/testing/negatives", 
						"data/svm_persondetect.xml");
	}

}
