package org.tribuo.classification.fs.wrapper.Main;

import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.fs.wrapper.Discreeting.TransferFunction;
import org.tribuo.classification.fs.wrapper.Evaluation.FitnessFunction;
import org.tribuo.classification.fs.wrapper.Optimizers.CuckooSearchOptimizer;
import org.tribuo.classification.sgd.fm.FMClassificationTrainer;
import org.tribuo.classification.sgd.linear.LinearSGDTrainer;
import org.tribuo.classification.sgd.objectives.Hinge;
import org.tribuo.classification.sgd.objectives.LogMulticlass;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.dataset.SelectedFeatureDataset;
import org.tribuo.evaluation.CrossValidation;
import org.tribuo.math.optimisers.AdaGrad;
import org.tribuo.util.Util;

import java.io.IOException;
import java.nio.file.Paths;

public class MainClass {
    public static void main(String... args) throws IOException {
        // read the data
        var dataPath = "...";
        var data = new CSVLoader<>(new LabelFactory()).loadDataSource(Paths.get(dataPath), "Class");
        var dataSet = new MutableDataset<>(data);

        // use the feature selection optimizer
        var optimizer = new CuckooSearchOptimizer(dataPath,
                FitnessFunction.Correlation_Id.PearsonsCorrelation,
                TransferFunction.V2,
                20,
                1.5d,
                2.5d,
                0.2d,
                1.5d,
                0.5,
                25,
                12345);
        
        var sDate = System.currentTimeMillis();
        var SFS = optimizer.select(dataSet);
        var eDate = System.currentTimeMillis();
        var SFDS = new SelectedFeatureDataset<>(dataSet, SFS);

        // use KNN classifier
        var KnnTrainer =  new KNNTrainer<>(3,
                new L1Distance(),
                Runtime.getRuntime().availableProcessors(),
                new VotingCombiner(),
                KNNModel.Backend.THREADPOOL,
                NeighboursQueryFactoryType.BRUTE_FORCE);

        // use crossvalidation
        // noinspection DuplicatedCode
        var crossValidation = new CrossValidation<>(KnnTrainer, SFDS, new LabelEvaluator(), 10, Trainer.DEFAULT_SEED);

        // get outputs
        var avgAcc = 0d;
        var sensitivity = 0d;
        var macroAveragedF1 = 0d;
        var sTrain = System.currentTimeMillis();
        for (var acc: crossValidation.evaluate()) {
            avgAcc += acc.getA().accuracy();
            sensitivity += acc.getA().tp() / (acc.getA().tp() + acc.getA().fn());
            macroAveragedF1 += acc.getA().macroAveragedF1();
        }
        var eTrain = System.currentTimeMillis();

        System.out.printf("The FS duration time is : %s\nThe number of selected features is : %d\nThe feature names are : %s\n",
                Util.formatDuration(sDate, eDate), SFS.featureNames().size(), SFS.featureNames());
        System.out.println("The Training_Testing duration time is : " + Util.formatDuration(sTrain, eTrain));
        System.out.println("The average accuracy is : " + (avgAcc / crossValidation.getK()));
        System.out.println("The average sensitivity is : " + (sensitivity / crossValidation.getK()));
        System.out.println("The average macroAveragedF1 is : " + (macroAveragedF1 / crossValidation.getK()));
        // store the resulted set of features
        System.out.println("Please Type Y to save the new set of features or N to ignore that");
        Scanner inputs = new Scanner(System.in);
        if (inputs.nextLine().equals("Y")) {
            System.out.println("Type the file name please");
            String dataName = inputs.nextLine();
            new CSVSaver().save(Paths.get("...\\" + dataName + ".csv"), SFDS, "Class");
        }
    }
}
