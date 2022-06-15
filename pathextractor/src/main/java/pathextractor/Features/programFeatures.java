package pathextractor.Features;

import java.util.ArrayList;
import java.util.HashMap;

import com.github.javaparser.ast.Node;

public class programFeatures {
    private ArrayList<MethodFeatures> programPaths;
    private HashMap<String, Node> methodMap;

    public programFeatures(ArrayList<MethodFeatures> programPaths, HashMap<String, Node> methodMap) {
        this.programPaths = programPaths;
        this.methodMap = methodMap;
    }

    public HashMap<String, Node> getMethodMap() {
        return methodMap;
    }

    public ArrayList<MethodFeatures> getProgramPaths() {
        return programPaths;
    }
}
