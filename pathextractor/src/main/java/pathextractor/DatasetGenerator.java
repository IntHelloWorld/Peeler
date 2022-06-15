package pathextractor;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.OverlappingFileLockException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;

import com.github.javaparser.ast.Node;

import pathextractor.Common.CommandLineValues;
import pathextractor.Common.ProgressBar;
import pathextractor.Features.MethodFeatures;
import pathextractor.Features.MethodPath;
import pathextractor.Features.ProgramRelation;
import pathextractor.Features.programFeatures;

public class DatasetGenerator {

    /**
     * Save extracted data to "corpus.txt".
     */
    public static void generateDataset(programFeatures programFeatures, CommandLineValues commandLineValues,
            Path classPath, String projectName, String label, ProgressBar progressBar) {
        Path corpusPath = Paths.get(commandLineValues.outputDir, "corpus.txt");
        // Path nameIdxsPath = Paths.get(commandLineValues.outputDir, "name_idxs.txt");
        // Path methodIdxsPath = Paths.get(commandLineValues.outputDir,
        // "method_idxs.txt");

        File corpus = new File(corpusPath.toString());
        // File nameIdxs = new File(nameIdxsPath.toString());
        // File methodIdxs = new File(methodIdxsPath.toString());
        FileChannel fc = null;
        FileLock fl = null;

        try {
            // Create a new file when it doesn't exist in mode "rw".
            RandomAccessFile corpusRaf = new RandomAccessFile(corpus, "rw");
            fc = corpusRaf.getChannel();
            boolean hasWrite = false;
            while (!hasWrite) {
                try {
                    fl = fc.tryLock();
                    if (fl != null) { // Get the file lock
                        int corpusIdx = readCorpusIndex(corpusRaf);
                        // Move pointers to the end of the files
                        corpusRaf.seek(corpusRaf.length());
                        // Write files
                        StringBuilder input = new StringBuilder();
                        if (programFeatures.getProgramPaths().size() == 0) {
                            input.append("\nWARNING :\n" + classPath.toString()
                                    + "\ndid not generate any program features !\n");
                        }
                        for (MethodFeatures methodFeatures : programFeatures.getProgramPaths()) {
                            HashMap<String, Node> methodMap = programFeatures.getMethodMap();
                            input.append("methods:\n");
                            input.append("class:" + classPath.toString() + '\n');
                            for (Entry<String, Node> entry : methodMap.entrySet()) {
                                input.append("name:" + entry.getKey() + '\n');
                                input.append("content:\n" + entry.getValue().toString() + "\nend_content\n");
                            }
                            input.append("end_methods\n\n");
                            input.append("begin_sample\nid:" + String.valueOf(corpusIdx + 1) + '\n');
                            input.append("method:" + methodFeatures.getName() + '\n');
                            input.append("label:" + label + "\n");
                            input.append("projectName:" + projectName + "\n");
                            input.append("class:" + classPath.toString() + '\n');
                            input.append("paths:\n");

                            LinkedList<MethodPath> methodPaths = methodFeatures.getPaths();
                            List<String> pathStrings = new ArrayList<String>();
                            for (MethodPath methodPath : methodPaths) {
                                LinkedList<ProgramRelation> path = methodPath.getPath();
                                StringBuilder pathString = new StringBuilder();
                                for (int i = 0; i < path.size(); i++) {
                                    ProgramRelation relation = path.get(i);
                                    if (i == 0) {
                                        pathString.append(
                                                relation.getSource().toString() + '\t' + relation.getEdge().toString()
                                                        + '\t' + relation.getTarget().toString());
                                    } else {
                                        pathString.append('\t' + relation.getEdge().toString() + '\t'
                                                + relation.getTarget().toString());
                                    }
                                }
                                if (!pathStrings.contains(pathString.toString())) {
                                    pathStrings.add(pathString.toString());
                                    input.append(pathString.toString().replace("\n", "") + "\n");
                                }
                            }
                            input.append("end_sample\n\n");
                            corpusIdx++;
                        }
                        progressBar.printProgress();
                        input.append("\n");
                        corpusRaf.write(input.toString().getBytes());
                        // Close files
                        corpusRaf.close();
                        hasWrite = true;
                    }
                } catch (OverlappingFileLockException oe) {
                    ;
                } catch (Exception e) {
                    e.printStackTrace(); // Waiting for the lock
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (fl != null && fl.isValid()) {
                try {
                    fl.release(); // Release file lock
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static int readIndex(RandomAccessFile raf) throws IOException {
        long len = raf.length();
        int res = -1;
        if (len != 0L) {
            long pos = len - 1;
            while (pos > 0) {
                pos--;
                raf.seek(pos);
                if (raf.readByte() == '\n') {
                    raf.seek(pos + 1);
                    res = raf.readInt();
                }
            }
        }
        return res;
    }

    public static int readCorpusIndex(RandomAccessFile raf) throws IOException {
        long len = raf.length();
        int res = -1;
        if (len != 0L) {
            long pos = len - 1;
            while (pos > 0) {
                pos--;
                raf.seek(pos);
                byte[] b = new byte[4];
                if (raf.read(b) == 4) {
                    String string = new String(b);
                    if (string.equals("\nid:")) {
                        pos++;
                        raf.seek(pos);
                        String idLine = raf.readLine();
                        String idString = idLine.substring(3);
                        int id = Integer.parseInt(idString);
                        res = id;
                        break;
                    }
                }
            }
        }
        return res;
    }
}
