package pathextractor.Features;

import java.util.LinkedList;

import com.fasterxml.jackson.annotation.JsonIgnore;

public class MethodFeatures {
    private String name;
    private LinkedList<MethodPath> paths;

    public MethodFeatures(String name, LinkedList<MethodPath> paths) {
        this.name = name;
        this.paths = paths;
    }

    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(name).append(" ");
        for (Object path : paths) {
            if (path instanceof String) {
                stringBuilder.append((String) path);
            } else {
                String edge = ((Edge) path).toString();
                stringBuilder.append(edge);
            }
        }
        return stringBuilder.toString();
    }

    public void addPath(MethodPath path) {
        paths.add(path);
    }

    @JsonIgnore
    public boolean isEmpty() {
        return paths.isEmpty();
    }

    public void deleteAllPaths() {
        paths.clear();
    }

    public String getName() {
        return name;
    }

    public LinkedList<MethodPath> getPaths() {
        return paths;
    }
}