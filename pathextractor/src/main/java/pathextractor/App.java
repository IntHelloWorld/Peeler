package pathextractor;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.nio.file.Path;
import java.util.Arrays;
// import java.util.LinkedList;
import java.util.List;
// import java.util.concurrent.Executors;
// import java.util.concurrent.ThreadPoolExecutor;
// import java.util.concurrent.locks.ReentrantLock;

import com.opencsv.CSVReader;

import org.apache.commons.exec.CommandLine;
import org.apache.commons.exec.DefaultExecutor;
import org.kohsuke.args4j.CmdLineException;

import pathextractor.Common.CommandLineValues;
import pathextractor.Common.Line;
import pathextractor.Common.ProgressBar;

public class App {
    private static CommandLineValues s_CommandLineValues;

    public static void main(String[] args) {
        try {
            s_CommandLineValues = new CommandLineValues(args);
        } catch (CmdLineException e) {
            e.printStackTrace();
            return;
        }

        if (s_CommandLineValues.projectDir != null && s_CommandLineValues.classPath != null
                && s_CommandLineValues.targetMethodNames != null && s_CommandLineValues.SHA != null) {
            String fileType = s_CommandLineValues.classPath.toString().split("\\.")[1];
            if (!fileType.equals("java")) {
                System.out.println("[WARNING] <project>: " + s_CommandLineValues.projectDir + " <class>: "
                        + s_CommandLineValues.classPath + " is not java file!");
                return;
            }
            List<String> targetMethodNames = Arrays.asList(s_CommandLineValues.targetMethodNames.split("\\."));
            Line line = new Line(s_CommandLineValues.classPath, targetMethodNames, s_CommandLineValues.label, "java");
            FeatureExtractTask featureExtractTask = new FeatureExtractTask(s_CommandLineValues, line,
                    new ProgressBar(1));
            featureExtractTask.processFile();
        } else if (s_CommandLineValues.csvFile != null && s_CommandLineValues.projectsDir != null) {
            solveAll(s_CommandLineValues.csvFile);
        } else {
            System.err.println("Command Line ERROR");
        }
    }

    private static void solveAll(File csvFile) {
        int lineNumber = getLineNumber(csvFile) - 2;
        ProgressBar progressBar = new ProgressBar(lineNumber);
        // ThreadPoolExecutor executor = (ThreadPoolExecutor)
        // Executors.newFixedThreadPool(s_CommandLineValues.NumThreads);
        // LinkedList<FeatureExtractTask> tasks = new LinkedList<>();
        try {
            CSVReader reader = new CSVReader(new InputStreamReader(new FileInputStream(csvFile)));
            String[] line = null;
            Line lastLine = null;
            reader.readNext();
            while ((line = reader.readNext()) != null) {
                Line currentLine = new Line(line);

                // Checkout the project.
                if (lastLine == null || (lastLine != null
                        && !(lastLine.projectName + lastLine.sha).equals(currentLine.projectName + currentLine.sha))) {
                    String cmd1 = "git checkout .";
                    String cmd2 = "git checkout " + currentLine.sha;
                    CommandLine cmdLine1 = CommandLine.parse(cmd1);
                    CommandLine cmdLine2 = CommandLine.parse(cmd2);
                    DefaultExecutor executor2 = new DefaultExecutor();
                    Path projectDir = s_CommandLineValues.projectsDir.resolve(currentLine.projectName);
                    s_CommandLineValues.projectDir = projectDir;
                    executor2.setWorkingDirectory(new File(s_CommandLineValues.projectDir.toString()));
                    executor2.execute(cmdLine1);
                    int exitValue = executor2.execute(cmdLine2);
                    if (exitValue != 0) {
                        System.err.println("Failed to checkout project: " + s_CommandLineValues.projectDir.toString());
                    }
                }

                if (!currentLine.fileType.equals("java")) {
                    System.out.println("[WARNING] <project>: " + currentLine.projectName + " <class>: "
                            + currentLine.classPath + " is not java file!");
                    lastLine = new Line(line);
                    continue;
                }

                FeatureExtractTask task = new FeatureExtractTask(s_CommandLineValues, currentLine, progressBar);
                task.processFile();
                // tasks.add(task);
                lastLine = new Line(line);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        // try {
        // executor.invokeAll(tasks);
        // } catch (InterruptedException e) {
        // e.printStackTrace();
        // } finally {
        // executor.shutdown();
        // }
    }

    public static int getLineNumber(File file) {
        if (file.exists()) {
            try {
                FileReader fileReader = new FileReader(file);
                LineNumberReader lineNumberReader = new LineNumberReader(fileReader);
                lineNumberReader.skip(Long.MAX_VALUE);
                int lines = lineNumberReader.getLineNumber() + 1;
                fileReader.close();
                lineNumberReader.close();
                return lines;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return 0;
    }
}
