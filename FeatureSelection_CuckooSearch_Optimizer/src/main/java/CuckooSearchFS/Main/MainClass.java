package CuckooSearchFS.Main;

import CuckooSearchFS.Discreeting.TransferFunction;
import CuckooSearchFS.Optimizers.CuckooSearchOptimizer;
import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.fs.wrapper.Evaluation.FitnessFunction;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.data.csv.CSVSaver;
import org.tribuo.dataset.SelectedFeatureDataset;
import org.tribuo.evaluation.CrossValidation;
import org.tribuo.interop.tensorflow.DenseFeatureConverter;
import org.tribuo.interop.tensorflow.GradientOptimiser;
import org.tribuo.interop.tensorflow.LabelConverter;
import org.tribuo.interop.tensorflow.TensorFlowTrainer;
import org.tribuo.interop.tensorflow.example.MLPExamples;
import org.tribuo.util.Util;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class MainClass {
    public static void main(String... args) throws IOException {
        // read the data
        var dataPath = "..."; // set your data path here
        var data = new CSVLoader<>(new LabelFactory()).loadDataSource(Paths.get(dataPath), "Class");
        var dataSet = new MutableDataset<>(data);

        // use the feature selection optimizer
        var optimizer = new CuckooSearchOptimizer(dataPath,
                FitnessFunction.Correlation_Id.PearsonsCorrelation,
                TransferFunction.V2,
                20,
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
        var NN_Graph = MLPExamples.buildMLPGraph("MLP_NN",
                SFDS.getFeatureMap().size(),
                new int[] {100, 50},
                SFDS.getOutputs().size());

        var optAlg = GradientOptimiser.GRADIENT_DESCENT;
        var optAlgParameters = Map.of("learningRate", 0.1f);

        var featuresType = new DenseFeatureConverter(NN_Graph.inputName);
        var labelConverter = new LabelConverter();

        var TF_Trainer = new TensorFlowTrainer<>(NN_Graph.graphDef,
                NN_Graph.outputName,
                optAlg,
                optAlgParameters,
                featuresType,
                labelConverter,
                16,
                30,
                16,
                -1);

        // use crossvalidation
        var crossValidation = new CrossValidation<>(TF_Trainer, SFDS, new LabelEvaluator(), 10, Trainer.DEFAULT_SEED);

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
        for (var stuff : List.of("The Training_Testing duration time is : " + Util.formatDuration(sTrain, eTrain),
                "The average accuracy is : " + (avgAcc / crossValidation.getK()),
                "The average sensitivity is : " + (sensitivity / crossValidation.getK()),
                "The average macroAveragedF1 is : " + (macroAveragedF1 / crossValidation.getK()))) {
            System.out.println(stuff);
        }
        // store the resulted set of features
        System.out.println("Please Type Y to save the new set of features or N to ignore that");
        var inputs = new Scanner(System.in);
        if (inputs.nextLine().equals("Y")) {
            var saveToThisPath = "...";  // set your path here
            var dataName = inputs.nextLine();
            new CSVSaver().save(Paths.get( saveToThisPath + dataName + ".csv"), SFDS, "Class");
        }
    }
}

