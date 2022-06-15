package pathextractor.Common;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class Line implements Cloneable {
    public String url;
    public String sha;
    public Path classPath;
    public List<String> targetMethodNames;
    public String projectName;
    public String label;
    public String fileType;

    public Line(String[] line) {
        projectName = line[0];
        sha = line[1];
        classPath = Paths.get(line[2]);
        targetMethodNames = Arrays.asList(line[3].split("\\."));
        label = line[4];
        fileType = classPath.getFileName().toString().split("\\.")[1];

        // String[] tmp = url.split("/");
        // projectName = tmp[tmp.length -1];
    }

    public Line(Path classPath, List<String> targetMethodNames, String label, String fileType) {
        this.classPath = classPath;
        this.targetMethodNames = targetMethodNames;
        this.label = label;
        this.fileType = fileType;
    }
}
