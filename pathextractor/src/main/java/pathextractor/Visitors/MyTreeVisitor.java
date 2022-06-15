package pathextractor.Visitors;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.expr.ArrayAccessExpr;
import com.github.javaparser.ast.expr.ArrayCreationExpr;
import com.github.javaparser.ast.expr.ArrayInitializerExpr;
import com.github.javaparser.ast.expr.AssignExpr;
import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.DoubleLiteralExpr;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.IntegerLiteralExpr;
import com.github.javaparser.ast.expr.LiteralExpr;
import com.github.javaparser.ast.expr.LongLiteralExpr;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.ast.expr.VariableDeclarationExpr;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import com.github.javaparser.ast.stmt.ForeachStmt;
import com.github.javaparser.ast.stmt.TryStmt;
import com.github.javaparser.ast.visitor.TreeVisitor;

import pathextractor.Common.Common;
import pathextractor.Features.Edge;
import pathextractor.Features.Element;
import pathextractor.Features.ProgramRelation;

public class MyTreeVisitor extends TreeVisitor {
    private ArrayList<ProgramRelation> relations = new ArrayList<>();
    private HashMap<String, String> typeMap = new HashMap<>();
    HashMap<String, Integer> valueMap = new HashMap<>();
    // Statements which conform to the pattern are regarded as assert statements.
    Pattern assertPattern = Pattern.compile(
            "(.*assert.*)|(.*mockServer.*)|(check.*\\()|(when\\()|(verify.+)|(logger)|(log)|(test.+)|(validate)|(mock)|(affirm)",
            Pattern.CASE_INSENSITIVE);

    @Override
    public void process(Node node) {
        if (node instanceof Comment) {
            return;

        } else if (isAssertExpr(node)) {
            MethodCallExpr mcNode = (MethodCallExpr) node;
            this.solveAssertExpr(mcNode, relations, Common.getBeginLineNumber(node));

        } else if (node instanceof MethodCallExpr && node.getParentNode() instanceof ExpressionStmt) {
            MethodCallExpr mcNode = (MethodCallExpr) node;
            this.solveMethodCallExpr(mcNode, relations, "NORMAL", Common.getBeginLineNumber(node));

        } else if (node instanceof VariableDeclarationExpr) {
            VariableDeclarationExpr vdeNode = (VariableDeclarationExpr) node;
            this.solveVariableDeclarationExpr(vdeNode, relations, Common.getBeginLineNumber(node));

        } else if (node instanceof FieldDeclaration) {
            FieldDeclaration fdNode = (FieldDeclaration) node;
            this.solveFieldDeclaration(fdNode, relations, Common.getBeginLineNumber(node));

        } else if (node instanceof AssignExpr) {
            AssignExpr assignNode = (AssignExpr) node;
            this.solveAssignExpr(assignNode, relations, Common.getBeginLineNumber(node));

        } else if (node instanceof ForeachStmt) {
            ForeachStmt foreachNode = (ForeachStmt) node;
            String targetName = foreachNode.getVariable().getVariables().get(0).toString();
            String sourceName = foreachNode.getIterable().toString();
            int lineNumber = Common.getBeginLineNumber(node);
            Edge edge = Common.EmptyEdge;
            ProgramRelation relation = new ProgramRelation(new Element(sourceName), edge, new Element(targetName),
                    "NORMAL", lineNumber);
            relations.add(relation);

            // Ignore finally block.
        } else if (node instanceof TryStmt) {
            TryStmt tryNode = (TryStmt) node;
            tryNode.setFinallyBlock(null);
        }

        // TODO : More kinds of nodes

    }

    private void solveAssertExpr(MethodCallExpr node, ArrayList<ProgramRelation> relations, int lineNumber) {
        String name = node.getName();

        // Collect all arguments in an assert statement.
        List<Expression> args = new ArrayList<>();
        for (Expression arg : node.getArgs()) {
            args.add(arg);
        }
        Set<Expression> binaryArgs = new HashSet<>();
        for (Expression arg : args) {
            if (arg instanceof BinaryExpr) {
                extractVariables((BinaryExpr) arg, binaryArgs);
            }
        }
        args.addAll(binaryArgs);
        List<Expression> scopeArgs = new ArrayList<>();
        if (node.getScope() instanceof MethodCallExpr) {
            MethodCallExpr scope = ((MethodCallExpr) node.getScope());
            scopeArgs = scope.getArgs();
        }
        args.addAll(scopeArgs);

        // Iteration and mark the MethodCallExpr within AssertExpr.
        for (Expression arg : args) {
            if (arg instanceof MethodCallExpr) {
                solveMethodCallExpr((MethodCallExpr) arg, relations, "ASSERT_MC", lineNumber);
            }
        }

        for (int i = 0; i < args.size(); i++) {
            Expression arg_i = args.get(i);
            for (int j = i; j < args.size(); j++) {
                Expression arg_j = args.get(j);
                Edge edge = new Edge(name, node.getScope() != null ? node.getScope().toString() : Common.EmptyString,
                        node.getArgs());
                ProgramRelation relation = new ProgramRelation(new Element(argToString(arg_i)), edge,
                        new Element(argToString(arg_j)), "ASSERT", lineNumber);
                relations.add(relation);
            }
        }
    }

    private void solveVariableDeclarationExpr(VariableDeclarationExpr node, ArrayList<ProgramRelation> relations,
            int lineNumber) {
        List<VariableDeclarator> variables = node.getVariables();
        String typeString = node.getElementType().toString();
        for (VariableDeclarator variable : variables) {
            String targetName = variable.getId().getName();
            Expression init = variable.getInit();
            typeMap.put(targetName, typeString);
            if (init == null) {
                String type = ((VariableDeclarationExpr) node).getElementType().toString();
                ProgramRelation relation = new ProgramRelation(new Element(type), Common.EmptyEdge,
                        new Element(targetName), "VARIABLE", lineNumber);
                relations.add(relation);
            } else {
                solveExprWithTarget(init, targetName, "VARIABLE", lineNumber);
            }
        }
    }

    private void solveFieldDeclaration(FieldDeclaration node, ArrayList<ProgramRelation> relations, int lineNumber) {
        List<VariableDeclarator> variables = node.getVariables();
        String typeString = node.getElementType().toString();
        for (VariableDeclarator variable : variables) {
            String targetName = variable.getId().getName();
            Expression init = variable.getInit();
            typeMap.put(targetName, typeString);
            if (init == null) {
                String type = ((FieldDeclaration) node).getElementType().toString();
                ProgramRelation relation = new ProgramRelation(new Element(type), Common.EmptyEdge,
                        new Element(targetName), "VARIABLE", lineNumber);
                relations.add(relation);
            } else {
                solveExprWithTarget(init, targetName, "VARIABLE", lineNumber);
            }
        }
    }

    private void solveAssignExpr(AssignExpr node, ArrayList<ProgramRelation> relations, int lineNumber) {
        String targetName = node.getTarget().toString();
        Expression value = node.getValue();
        solveExprWithTarget(value, targetName, "VARIABLE", lineNumber);
    }

    /**
     * Extract relations from Expression when given targetName.
     */
    private void solveExprWithTarget(Expression expr, String targetName, String type, int lineNumber) {
        if (expr instanceof MethodCallExpr) {
            solveMethodCallExpr((MethodCallExpr) expr, relations, "MC_" + type, lineNumber);
            String sourceName = exprToString(((MethodCallExpr) expr));
            ProgramRelation relation = new ProgramRelation(new Element(sourceName), Common.EmptyEdge,
                    new Element(targetName), type, lineNumber);
            relations.add(relation);

        } else if (expr instanceof ObjectCreationExpr) {
            // TODO: if we need change the edge type? AnonymousClassBody is ignored here.
            ObjectCreationExpr ocExpr = (ObjectCreationExpr) expr;
            String sourceName = (ocExpr).getType().toString();
            ProgramRelation relation = new ProgramRelation(new Element(sourceName), Common.EmptyEdge,
                    new Element(targetName), type, lineNumber);
            relations.add(relation);

        } else if (expr instanceof BinaryExpr) {
            Set<Expression> binaryArgs = new HashSet<>();
            extractVariables((BinaryExpr) expr, binaryArgs);
            if (binaryArgs.size() > 0) {
                for (Expression arg : binaryArgs) {
                    String sourceName = exprToString(arg);
                    ProgramRelation relation = new ProgramRelation(new Element(sourceName), Common.EmptyEdge,
                            new Element(targetName), type, lineNumber);
                    relations.add(relation);
                }
            } else {
                String sourceName = "LiteralExpression";
                ProgramRelation relation = new ProgramRelation(new Element(sourceName), Common.EmptyEdge,
                        new Element(targetName), type, lineNumber);
                relations.add(relation);
            }

        } else if (expr instanceof ArrayInitializerExpr) {
            List<Expression> values = ((ArrayInitializerExpr) expr).getValues();
            for (Expression value : values) {
                solveExprWithTarget(value, targetName, type, lineNumber);
            }

        } else {
            ProgramRelation relation = new ProgramRelation(new Element(Common.getClassName(expr.getClass())),
                    Common.EmptyEdge, new Element(targetName), type, lineNumber);
            relations.add(relation);
        }
        // TODO: more kinds of init expr.
    }

    /**
     * Consider all the possible data relation in a MethodCallExpr.
     */
    private void solveMethodCallExpr(MethodCallExpr node, ArrayList<ProgramRelation> relations, String assertLabel,
            int lineNumber) {
        boolean hasScope = false;
        if (node.getScope() != null) {
            hasScope = true;
        }

        if (hasScope && node.getScope() instanceof MethodCallExpr) {
            solveMethodCallExpr((MethodCallExpr) node.getScope(), relations, assertLabel, lineNumber);
        }
        String name = node.getName();
        if (name.length() > 5 && name.substring(0, 5).equals("print")) { // Ignore output stmt.
            return;
        }
        List<Expression> args = node.getArgs();
        for (Expression arg : args) {
            if (arg instanceof MethodCallExpr) {
                solveMethodCallExpr((MethodCallExpr) arg, relations, assertLabel, lineNumber);
            }
        }
        String scope = hasScope ? exprToString(node.getScope()) : "null";
        String nodeString = exprToString(node);
        Edge edge = new Edge(name, scope, args);

        ArrayList<String> sourceNode = new ArrayList<String>() {
            private static final long serialVersionUID = 1L;
            {
                add(scope);
                for (Expression arg : args) {
                    add(exprToString(arg));
                }
            }
        };

        ArrayList<String> targetNode = new ArrayList<String>() {
            private static final long serialVersionUID = 1L;
            {
                add(scope);
                for (Expression arg : args) {
                    add(exprToString(arg));
                }
                add(nodeString);
            }
        };

        for (String source : sourceNode) {
            for (String target : targetNode) {
                ProgramRelation relation = new ProgramRelation(new Element(source), edge, new Element(target),
                        assertLabel, lineNumber);
                relations.add(relation);
            }
        }
    }

    private boolean isAssertExpr(Node node) {
        if (node instanceof MethodCallExpr) {
            String name = ((MethodCallExpr) node).getName();
            if (((MethodCallExpr) node).getScope() != null) {
                String scopeString = ((MethodCallExpr) node).getScope().toString();
                Matcher m1 = assertPattern.matcher(scopeString);
                if (m1.find())
                    return true;
            }
            Matcher m0 = assertPattern.matcher(name);
            if (m0.find()) {
                return true;
            }
        }
        return false;
    }

    private void extractVariables(BinaryExpr expr, Set<Expression> binaryArgs) {
        Expression right = expr.getRight();
        Expression left = expr.getLeft();
        if (expr.getLeft() instanceof BinaryExpr) {
            if (!(right instanceof LiteralExpr)) {
                binaryArgs.add(right);
            }
            extractVariables((BinaryExpr) expr.getLeft(), binaryArgs);
        } else {
            if (!(right instanceof LiteralExpr)) {
                binaryArgs.add(right);
            }
            if (!(left instanceof LiteralExpr)) {
                binaryArgs.add(left);
            }
        }
    }

    private String exprToString(Expression expr) {
        if (expr instanceof MethodCallExpr) {
            MethodCallExpr mcExpr = (MethodCallExpr) expr;
            StringBuilder stringBuilder = new StringBuilder();
            Expression scope = mcExpr.getScope();
            if (scope != null) {
                stringBuilder.append(exprToString(scope) + ".");
            }
            stringBuilder.append(mcExpr.getName() + "(");
            for (Expression arg : mcExpr.getArgs()) {
                stringBuilder.append(exprToString(arg) + ",");
            }
            stringBuilder.deleteCharAt(stringBuilder.length() - 1);
            stringBuilder.append(")");
            return stringBuilder.toString();

        } else if (expr instanceof IntegerLiteralExpr || expr instanceof LongLiteralExpr
                || expr instanceof DoubleLiteralExpr) {
            String key = expr.toString();
            if (!valueMap.containsKey(key)) {
                valueMap.put(key, valueMap.size());
            }
            return "value" + Integer.toString(valueMap.get(key));

        } else if (expr instanceof BinaryExpr) {
            BinaryExpr bExpr = ((BinaryExpr) expr);
            Expression right = bExpr.getRight();
            Expression left = bExpr.getLeft();
            return exprToString(left) + bExpr.getOperator().toString() + exprToString(right);

        } else if (expr instanceof ArrayAccessExpr) {
            String arrayName = ((ArrayAccessExpr) expr).getName().toString();
            return arrayName;

        } else if (expr instanceof ArrayCreationExpr) {
            String arrayName = ((ArrayCreationExpr) expr).getType().toString();
            return arrayName;

        } else {
            return expr.toString();
        }
    }

    private String argToString(Expression arg) {
        if (arg instanceof LiteralExpr || arg instanceof BinaryExpr) {
            return Common.getClassName(arg.getClass());

        } else if (arg instanceof ArrayAccessExpr) {
            String arrayName = ((ArrayAccessExpr) arg).getName().toString();
            return arrayName;

        } else if (arg instanceof ArrayCreationExpr) {
            String arrayName = ((ArrayCreationExpr) arg).getType().toString();
            return arrayName;

        } else {
            return arg.toString();
        }
    }

    public ArrayList<ProgramRelation> getProgramRelations() {
        return relations;
    }

    public HashMap<String, String> getTypeMap() {
        return typeMap;
    }
}
