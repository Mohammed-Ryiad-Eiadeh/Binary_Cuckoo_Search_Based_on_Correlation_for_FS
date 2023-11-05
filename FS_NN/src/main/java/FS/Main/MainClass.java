package FS.Main;

import FS.Discreeting.TransferFunction;
import FS.Optimizers.CuckooSearchOptimizer;
import org.tribuo.MutableDataset;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.data.csv.CSVLoader;
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
import java.util.Map;

public class MainClass {
    public static void main(String... args) throws IOException {
        // read the data
        var dataPath = "...";  // the path of your dataset
        var data = new CSVLoader<>(new LabelFactory()).loadDataSource(Paths.get(dataPath), "Class");
        var dataSet = new MutableDataset<>(data);

        var optimizer = new CuckooSearchOptimizer(TransferFunction.V2,
                20,
                2,
                2,
                0.1,
                1.5,
                0.1,
                10,
                12345);

        var sDate = System.currentTimeMillis();
        var SFS = optimizer.select(dataSet);
        var eDate = System.currentTimeMillis();
        var SFDS = new SelectedFeatureDataset<>(dataSet, SFS);

        // use FM classifier
        var NN = MLPExamples.buildMLPGraph("NN",
                SFDS.getFeatureMap().size(),
                new int[] {100, 50},
                SFDS.getOutputs().size());

        var opt = GradientOptimiser.GRADIENT_DESCENT;
        var optParameters = Map.of("learningRate", 0.1f);

        var featureConverter = new DenseFeatureConverter(NN.inputName);
        var outputConverter = new LabelConverter();

        var TF_NN = new TensorFlowTrainer<>(NN.graphDef,
                NN.outputName,
                opt,
                optParameters,
                featureConverter,
                outputConverter,
                16,
                30,
                16,
                -1);

        // use crossvalidation
        var crossValidation = new CrossValidation<>(TF_NN, SFDS, new LabelEvaluator(), 10);

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

