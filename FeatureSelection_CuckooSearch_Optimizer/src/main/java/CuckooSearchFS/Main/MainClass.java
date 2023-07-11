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
        var dataPath = "D:\\Mohammeds\\Mohammeds Work\\my master projects\\CS thesis\\datasets\\data sets\\sonar dataset\\sonar.csv";
        var data = new CSVLoader<>(new LabelFactory()).loadDataSource(Paths.get(dataPath), "Class");
        var dataSet = new MutableDataset<>(data);

        // use the feature selection optimizer based on the given learner
        var learner = new LinearSGDTrainer(new LogMulticlass(),
                new AdaGrad(0.1f, 0.6f),
                100,
                Trainer.DEFAULT_SEED);

        var optimizer = new CuckooSearchOptimizer(dataPath,
                learner,
                FitnessFunction.Correlation_Id.PearsonsCorrelation,
                TransferFunction.V2,
                20,
                1.5d,
                2.5d,
                0.2d,
                1.5d,
                2,
                12345);

        var sDate = System.currentTimeMillis();
        var SFS = optimizer.select(dataSet);
        var eDate = System.currentTimeMillis();
        var SFDS = new SelectedFeatureDataset<>(dataSet, SFS);

        // use FM classifier
        var FMTrainer = new FMClassificationTrainer(new Hinge(),
                new AdaGrad(0.1, 0.8),
                100,
                Trainer.DEFAULT_SEED,
                10,
                0.2D);

        // use crossvalidation
        var crossValidation = new CrossValidation<>(FMTrainer, SFDS, new LabelEvaluator(), 2);

        // get outputs
        var avgAcc = 0D;
        var sTrain = System.currentTimeMillis();
        for (var acc: crossValidation.evaluate())
            avgAcc += acc.getA().accuracy();
        var eTrain = System.currentTimeMillis();

        System.out.printf("The FS duration time is : %s\nThe number of selected features is : %d\nThe feature names are : %s\n",
                Util.formatDuration(sDate, eDate), SFS.featureNames().size(), SFS.featureNames());

        System.out.println("The Training_Testing duration time is : " + Util.formatDuration(sTrain, eTrain));
        System.out.println("The average accuracy is : " + (avgAcc / crossValidation.getK()));
    }
}
