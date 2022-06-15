package pathextractor.Visitors;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import pathextractor.Common.MethodContent;
import pathextractor.Features.ProgramRelation;

/**
 * Visit MethodDeclaration and code with @Rule annotation, collect data
 * relations and method information.
 * 
 * @author Yihao Qin
 */
public class FunctionVisitor extends VoidVisitorAdapter<Object> {
    private ArrayList<MethodContent> m_Methods = new ArrayList<>();
    private HashMap<String, String> typeMap = new HashMap<>();
    ArrayList<ProgramRelation> fieldRelations = new ArrayList<>();
    ArrayList<ProgramRelation> relations = new ArrayList<>();
    public List<String> targetMethodNames;

    public FunctionVisitor(List<String> targetMethodNames) {
        this.targetMethodNames = targetMethodNames;
    }

    // Collect relations from field declarations of the test case class
    @Override
    public void visit(FieldDeclaration node, Object arg) {
        visitMethod(node, arg);
        super.visit(node, arg);
    }

    // Collect relations from the body of test methods
    @Override
    public void visit(MethodDeclaration node, Object arg) {
        if (targetMethodNames.contains(node.getName())) {
            visitMethod(node, arg);
        }
        super.visit(node, arg);
    }

    private void visitMethod(FieldDeclaration node, Object obj) {
        MyTreeVisitor myTreeVisitor = new MyTreeVisitor();
        myTreeVisitor.visitDepthFirst(node);
        fieldRelations.addAll(myTreeVisitor.getProgramRelations());
        typeMap.putAll(myTreeVisitor.getTypeMap());
    }

    private void visitMethod(MethodDeclaration node, Object obj) {
        MyTreeVisitor myTreeVisitor = new MyTreeVisitor();
        myTreeVisitor.visitDepthFirst(node);
        // For each method declaration, clear relations and re-add the field relations
        relations.clear();
        relations.addAll(fieldRelations);
        relations.addAll(myTreeVisitor.getProgramRelations());
        typeMap.putAll(myTreeVisitor.getTypeMap());

        if (node.getBody() != null) {
            m_Methods.add(new MethodContent(relations, node.getName(), getMethodLength(node.getBody().toString())));
        }
    }

    private long getMethodLength(String code) {
        String cleanCode = code.replaceAll("\r\n", "\n").replaceAll("\t", " ");
        if (cleanCode.startsWith("{\n"))
            cleanCode = cleanCode.substring(3).trim();
        if (cleanCode.endsWith("\n}"))
            cleanCode = cleanCode.substring(0, cleanCode.length() - 2).trim();
        if (cleanCode.length() == 0) {
            return 0;
        }
        long codeLength = Arrays.asList(cleanCode.split("\n")).stream()
                .filter(line -> (line.trim() != "{" && line.trim() != "}" && line.trim() != ""))
                .filter(line -> !line.trim().startsWith("/") && !line.trim().startsWith("*")).count();
        return codeLength;
    }

    public ArrayList<MethodContent> getMethodContents() {
        return m_Methods;
    }

    public HashMap<String, String> getTypeMap() {
        return typeMap;
    }
}
