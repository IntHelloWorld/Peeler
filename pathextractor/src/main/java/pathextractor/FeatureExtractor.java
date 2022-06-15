package pathextractor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;

import pathextractor.Features.Element;

import com.github.javaparser.ParseException;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.MethodDeclaration;

import pathextractor.Common.CommandLineValues;
import pathextractor.Common.Common;
import pathextractor.Common.MethodContent;
import pathextractor.Features.Edge;
import pathextractor.Features.MethodFeatures;
import pathextractor.Features.MethodPath;
import pathextractor.Features.ProgramRelation;
import pathextractor.Features.programFeatures;
import pathextractor.Visitors.FunctionVisitor;
import pathextractor.Visitors.MethodDeclarationVisitor;

public class FeatureExtractor {
    private CommandLineValues m_CommandLineValues;

    public FeatureExtractor(CommandLineValues commandLineValues) {
        this.m_CommandLineValues = commandLineValues;
    }

    public programFeatures extractFeatures(String code, Path filePath, List<String> targetMethodNames)
            throws ParseException, IOException {
        CompilationUnit compilationUnit = Common.parseFileWithRetries(code);

        System.out.print("Extracting: " + filePath.toString() + targetMethodNames.toString() + '\n');
        // System.out.println("Extract program relations");
        FunctionVisitor functionVisitor = new FunctionVisitor(targetMethodNames);
        functionVisitor.visit(compilationUnit, null);
        ArrayList<MethodContent> methods = functionVisitor.getMethodContents();
        // System.out.println("Extract program paths");
        ArrayList<MethodFeatures> program = generateProgramFeatures(methods);
        // System.out.println("Extract program typeMap : variableName -> typeName");
        HashMap<String, String> typeMap = functionVisitor.getTypeMap();
        // System.out.println("Collect java file paths of import classes");
        ArrayList<Path> importClassPaths = ImportClassExtractor.extractImportClassPaths(compilationUnit, filePath);
        // System.out.println("Collect all the edges in program paths");
        HashSet<Edge> edgeSet = extractEdges(program);
        // System.out.println("Extract program map : methodName -> methodDeclaration");
        HashMap<String, Node> methodMap = generateMethodMap(importClassPaths, edgeSet, typeMap, compilationUnit);

        programFeatures programFeatures = new programFeatures(program, methodMap);
        return programFeatures;
    }

    private HashSet<Edge> extractEdges(ArrayList<MethodFeatures> program) {
        HashSet<Edge> edgeSet = new HashSet<>();
        for (MethodFeatures methodFeatures : program) {
            for (MethodPath methodPath : methodFeatures.getPaths()) {
                for (ProgramRelation relation : methodPath.getPath()) {
                    Edge edge = relation.getEdge();
                    if (!edgeSet.contains(edge)) {
                        edgeSet.add(edge);
                    }
                }
            }
        }
        return edgeSet;
    }

    private HashMap<String, Node> generateMethodMap(ArrayList<Path> importClassPaths, HashSet<Edge> edgeSet,
            HashMap<String, String> typeMap, CompilationUnit compilationUnit) throws IOException {
        HashMap<String, Node> methodMap = new HashMap<>();
        for (Edge edge : edgeSet) {
            MethodDeclarationVisitor methodDeclarationVisitor = new MethodDeclarationVisitor(edge.getName());
            // extract method map from import files.
            for (Path importClassPath : importClassPaths) {
                String importClassName = importClassPath.getFileName().toString().replace(".java", "");
                String callerName = edge.getCallerName();
                if (callerName.equals(importClassName)
                        || (typeMap.containsKey(callerName) ? typeMap.get(callerName).equals(importClassName)
                                : false)) {
                    String code;
                    try {
                        code = new String(Files.readAllBytes(importClassPath));
                    } catch (IOException e) {
                        e.printStackTrace();
                        code = Common.EmptyString;
                    }
                    CompilationUnit importFileCompilationUnit = Common.parseFileWithRetries(code);
                    methodDeclarationVisitor.visit(importFileCompilationUnit, null);
                    MethodDeclaration mcNode = methodDeclarationVisitor.getMethodDeclarationNode();
                    if (mcNode != null)
                        methodMap.put(edge.getName(), mcNode);
                }
            }
            // extract method map from local java test file.
            methodDeclarationVisitor.visit(compilationUnit, null);
            MethodDeclaration mcNode = methodDeclarationVisitor.getMethodDeclarationNode();
            if (mcNode != null)
                methodMap.put(edge.getName(), mcNode);
        }
        return methodMap;
    }

    private ArrayList<MethodFeatures> generateProgramFeatures(ArrayList<MethodContent> methods) {
        ArrayList<MethodFeatures> programFeatures = new ArrayList<MethodFeatures>();
        for (MethodContent methodContent : methods) {
            if (methodContent.getLength() < m_CommandLineValues.MinCodeLength
                    || methodContent.getLength() > m_CommandLineValues.MaxCodeLength)
                continue;
            LinkedList<MethodPath> paths = new LinkedList<MethodPath>();
            ArrayList<ProgramRelation> relations = methodContent.getRelations();
            for (ProgramRelation relation : relations) {
                String type = relation.type;
                if (type.equals("VARIABLE") || type.equals("ASSERT")) {
                    List<MethodPath> tmpPath = extractPaths(relation, relations);
                    if (tmpPath != null)
                        paths.addAll(tmpPath);
                }
            }
            // Control gross paths amount below 1000
            if (paths.size() > 1000) {
                for (int j = 0; j < paths.size() - 1000; j++)
                    paths.removeLast();
            }
            MethodFeatures methodFeatures = new MethodFeatures(methodContent.getName(), paths);
            programFeatures.add(methodFeatures);
        }
        return programFeatures;
    }

    @SuppressWarnings("unchecked")
    private LinkedList<MethodPath> visitMethodPaths(LinkedList<MethodPath> methodPaths, ProgramRelation relation) {
        LinkedList<MethodPath> newMethodPaths = (LinkedList<MethodPath>) methodPaths.clone();
        for (MethodPath methodPath : methodPaths) {
            if (methodPath.path.size() <= m_CommandLineValues.MaxPathLength) {
                if (relation.lineNumber >= methodPath.getPath().getLast().lineNumber) {
                    MethodPath extendedPath = MethodPath.getExtendedPath(relation, methodPath);
                    if (extendedPath != null && !newMethodPaths.contains(extendedPath)) {
                        newMethodPaths.add(extendedPath);
                    }
                }
            }
        }
        return newMethodPaths;
    }

    private List<MethodPath> extractPaths(ProgramRelation beginRelation, ArrayList<ProgramRelation> relations) {
        LinkedList<MethodPath> methodPaths = new LinkedList<MethodPath>();
        methodPaths.add(new MethodPath(beginRelation));

        for (int i = 0; i < relations.size(); i++) {
            ProgramRelation relation = relations.get(i);
            if (relation.lineNumber >= beginRelation.lineNumber)
                methodPaths = visitMethodPaths(methodPaths, relation);
            // TODO : Limit the size of methodPaths
            if (methodPaths.size() > m_CommandLineValues.MaxPathCount) {
                for (int j = 0; j < methodPaths.size() - m_CommandLineValues.MaxPathCount; j++)
                    methodPaths.removeFirst();
            }
        }

        // TODO : The alternative approach to choose the generated paths.
        return selectPaths(methodPaths);
    }

    /**
     * Algorithm to select representative paths of the generated paths. For each
     * beginRelation, select out all the ASSERT paths and all the longest path.
     */
    private List<MethodPath> selectPaths(List<MethodPath> methodPaths) {
        List<MethodPath> selectedPaths = new ArrayList<>();
        HashSet<Element> assertTargetSet = new HashSet<>();
        int longestSize = 0;
        for (MethodPath methodPath : methodPaths) {
            ProgramRelation lastRelation = methodPath.path.getLast();
            if (lastRelation.type.equals("ASSERT")) {
                if (!assertTargetSet.contains(lastRelation.getTarget())) {
                    selectedPaths.add(methodPath);
                    assertTargetSet.add(lastRelation.getTarget());
                }
            } else {
                int size = methodPath.getPath().size();
                if (size >= longestSize) {
                    longestSize = size;
                }
            }
        }
        for (MethodPath methodPath : methodPaths) {
            if (methodPath.getPath().size() == longestSize)
                selectedPaths.add(methodPath);
        }
        return selectedPaths;
    }
}
