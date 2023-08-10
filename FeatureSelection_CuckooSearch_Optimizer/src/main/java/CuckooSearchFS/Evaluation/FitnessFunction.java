package org.tribuo.classification.fs.wrapper.Evaluation;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.apache.commons.math3.stat.correlation.KendallsCorrelation;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;
import org.tribuo.*;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.dataset.SelectedFeatureDataset;
import org.tribuo.evaluation.CrossValidation;
import org.tribuo.provenance.FeatureSetProvenance;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class FitnessFunction {
    private final Trainer<Label> trainer;
    static double[][] matrix = null;
    private Correlation_Id correlation_id;

    /**
     * This interface includes the evaluation function of each solution
     */
    public enum Correlation_Id {
        PearsonsCorrelation, SpearmansCorrelation, KendallsCorrelation
    }

    /**
     * Default constructor of the fitness function utility
     * @param trainer The used trainer in the evaluation process
     */
    public FitnessFunction(Trainer<Label> trainer) {
        this.trainer = trainer;
    }

    /**
     * Constructs the matrix of the entire data in order to make it suitable to compute the correlation
     * @param path The URL of the data as a .CSV file format to be converted to matrix
     * @param trainer The used trainer in the evaluation process
     * @param c_id The used correlation function in the evaluation process
     */
    public FitnessFunction(String path, Trainer<Label> trainer, Correlation_Id c_id) {
        this.trainer = trainer;
        matrix = getMatrix(path);
        this.correlation_id = c_id;
    }

    /**
     * This method is used to compute the fitness score of each solution of the population
     * @param optimizer The optimizer that is used for FS
     * @param dataset The dataset to use
     * @param Fmap The dataset feature map
     * @param solution The current subset of features
     * @return The fitness score of the given subset
     */
    public <T extends FeatureSelector<Label>> double EvaluateSolution(T optimizer, Dataset<Label> dataset, ImmutableFeatureMap Fmap, int... solution) {
        SelectedFeatureDataset<Label> selectedFeatureDataset = new SelectedFeatureDataset<>(dataset,getSFS(optimizer, dataset, Fmap, solution));
        CrossValidation<Label, LabelEvaluation> crossValidation = new CrossValidation<>(trainer, selectedFeatureDataset, new LabelEvaluator(), 10);
        double avgAccuracy = 0D;
        for (Pair<LabelEvaluation, Model<Label>> ACC : crossValidation.evaluate()) {
            avgAccuracy += ACC.getA().accuracy();
        }
        avgAccuracy /= crossValidation.getK();
        double correlation = 0;
        if (matrix != null) {
            correlation = getCorrelation(correlation_id, solution);
        }
        return avgAccuracy + 0.001 * (1 - ((double) selectedFeatureDataset.getSelectedFeatures().size() / Fmap.size()) - correlation);
    }

    /**
     * This methid is used to return the selected subset of features
     * @param optimizer The optimizer that is used for FS
     * @param dataset The dataset to use
     * @param Fmap The dataset feature map
     * @param solution The current subset of featurs
     * @return The selected feature set
     */
    public <T extends FeatureSelector<Label>> SelectedFeatureSet getSFS(T optimizer, Dataset<Label> dataset, ImmutableFeatureMap Fmap, int... solution) {
        List<String> names = new ArrayList<>();
        List<Double> scores = new ArrayList<>();
        for (int i = 0; i < solution.length; i++) {
            if (solution[i] == 1) {
                names.add(Fmap.get(i).getName());
                scores.add(1D);
            }
        }
        FeatureSetProvenance provenance = new FeatureSetProvenance(SelectedFeatureSet.class.getName(), dataset.getProvenance(), optimizer.getProvenance());
        return new SelectedFeatureSet(names, scores, optimizer.isOrdered(), provenance);
    }

    /**
     * This method is used to convert the given data (.csv) into two dimention matrix
     * @param dataPath The URL of the .CSV data file
     * @return The 2D matrix of the given data
     */
    private double[][] getMatrix(String dataPath) {
        double[][] matrix = null;
        try (Scanner originalData = new Scanner(new File(dataPath))) {
            originalData.nextLine();
            List<String> listOfRows = new ArrayList<>();
            while (originalData.hasNext()) {
                listOfRows.add(originalData.nextLine());
            }

            matrix = new double[listOfRows.size()][listOfRows.get(0).split(",").length - 1];
            for (int i = 0; i < listOfRows.size(); i++) {
                String[] breakLine = listOfRows.get(i).split(",");
                for (int ii = 0; ii < breakLine.length - 1; ii++) {
                    matrix[i][ii] = Double.parseDouble(breakLine[ii]);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return matrix;
    }

    /**
     * This method is used to compute the correlation value of the given solution to check the dependency ratio between variables
     * @param correlation_id The function to be used to compute the correlation
     * @param solution The current solution from the current generation
     * @return The correlation value of the given solution
     */
    private double getCorrelation(Correlation_Id correlation_id, int... solution) {
        List<Integer> indecies = new ArrayList<>();
        for (int index = 0; index < solution.length; index++) {
            if (solution[index] == 1)
                indecies.add(index);
        }
        double[][] mat = new double[matrix.length][indecies.size() - 1];
        for (int r = 0; r < mat.length; r++) {
            for (int c = 0; c < mat[0].length; c++) {
                mat[r][c] = matrix[r][indecies.get(c)];
            }
        }
        return switch (correlation_id) {
            case PearsonsCorrelation -> new PearsonsCorrelation().computeCorrelationMatrix(mat).getNorm() / mat[0].length;
            case SpearmansCorrelation -> new SpearmansCorrelation().computeCorrelationMatrix(mat).getNorm() / mat[0].length;
            case KendallsCorrelation -> new KendallsCorrelation().computeCorrelationMatrix(mat).getNorm() / mat[0].length;
        };
    }
}

