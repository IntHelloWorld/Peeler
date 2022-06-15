package pathextractor;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

import org.apache.commons.lang3.StringUtils;

import com.github.javaparser.ParseException;

import pathextractor.Common.CommandLineValues;
import pathextractor.Common.Common;
import pathextractor.Common.Line;
import pathextractor.Common.ProgressBar;
import pathextractor.Features.MethodFeatures;
import pathextractor.Features.programFeatures;

public class FeatureExtractTask implements Callable<Void> {
    CommandLineValues m_CommandLineValues;
    Path filePath;
    Path classPath;
    String projectName;
    String label;
    List<String> methodNames;
    ProgressBar progressBar = null;

    public FeatureExtractTask(CommandLineValues commandLineValues, Line currentLine, ProgressBar progressBar) {
        m_CommandLineValues = commandLineValues;
        this.classPath = currentLine.classPath;
        this.projectName = currentLine.projectName;
        this.filePath = m_CommandLineValues.projectDir.resolve(classPath);
        this.label = currentLine.label;
        this.methodNames = currentLine.targetMethodNames;
        this.progressBar = progressBar;
    }

    @Override
    public Void call() throws Exception {
        processFile();
        return null;
    }

    public void processFile() {
        programFeatures features;
        try {
            features = extractSingleFile();
            // System.out.println("GenerateDataset and save to txt");
            DatasetGenerator.generateDataset(features, m_CommandLineValues, classPath, projectName, label, progressBar);
        } catch (ParseException | IOException e) {
            e.printStackTrace();
            return;
        }
        if (features == null) {
            return;
        }
    }

    public void SaveFeaturesToFile(Path filename, String content) {
        try {
            FileWriter fw = new FileWriter(filename.toString(), false);
            fw.write(content);
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public programFeatures extractSingleFile() throws ParseException, IOException {
        String code = null;
        try {
            code = new String(Files.readAllBytes(this.filePath));
        } catch (IOException e) {
            e.printStackTrace();
            code = Common.EmptyString;
        }
        FeatureExtractor featureExtractor = new FeatureExtractor(m_CommandLineValues);
        programFeatures programFeatures = featureExtractor.extractFeatures(code, this.filePath, this.methodNames);
        return programFeatures;

    }

    public String featuresToString(ArrayList<MethodFeatures> features) {
        if (features == null || features.isEmpty()) {
            return Common.EmptyString;
        }

        List<String> methodsOutputs = new ArrayList<>();

        for (MethodFeatures singleMethodfeatures : features) {
            StringBuilder builder = new StringBuilder();

            String output = Common.EmptyString;
            output = singleMethodfeatures.toString();
            if (m_CommandLineValues.PrettyPrint) {
                output = output.replace(" ", "\n\t");
            }
            builder.append(output);

            methodsOutputs.add(builder.toString());

        }
        return StringUtils.join(methodsOutputs, "\n");
    }
}
