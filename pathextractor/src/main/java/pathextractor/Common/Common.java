package pathextractor.Common;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.UserDataKey;
import com.github.javaparser.ast.expr.Expression;

import pathextractor.Features.Edge;

public final class Common {
    public static final UserDataKey<Integer> ChildId = new UserDataKey<Integer>() {
    };
    public static final String EmptyString = "";
    public static final List<Expression> EmptyExprList = new ArrayList<Expression>();
    public static final Edge EmptyEdge = new Edge(EmptyString, EmptyString, EmptyExprList);
    public static final String UTF8 = "UTF-8";
    public static final String EvaluateTempDir = "EvalTemp";

    public static final String FieldAccessExpr = "FieldAccessExpr";
    public static final String ClassOrInterfaceType = "ClassOrInterfaceType";
    public static final String MethodDeclaration = "MethodDeclaration";
    public static final String NameExpr = "NameExpr";
    public static final String MethodCallExpr = "MethodCallExpr";
    public static final String DummyNode = "DummyNode";
    public static final String BlankWord = "BLANK";

    public static final String EndNode = "END";
    public static final String TrueNode = "TRUE";
    public static final String FalseNode = "FALSE";

    public static final int c_MaxLabelLength = 50;
    public static final String methodName = "METHOD_NAME";
    public static final String internalSeparator = "|";

    public static String normalizeName(String original, String defaultString) {
        original = original.toLowerCase().replaceAll("\\\\n", "") // escaped new
                                                                  // lines
                .replaceAll("//s+", "") // whitespaces
                .replaceAll("[\"',]", "") // quotes, apostrophies, commas
                .replaceAll("\\P{Print}", ""); // unicode weird characters
        String stripped = original.replaceAll("[^A-Za-z]", "");
        if (stripped.length() == 0) {
            String carefulStripped = original.replaceAll(" ", "_");
            if (carefulStripped.length() == 0) {
                return defaultString;
            } else {
                return carefulStripped;
            }
        } else {
            return stripped;
        }
    }

    public static ArrayList<String> splitToSubtokens(String str1) {
        String str2 = str1.trim();
        return Stream.of(str2.split("(?<=[a-z])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\\s+"))
                .filter(s -> s.length() > 0).map(s -> Common.normalizeName(s, Common.EmptyString))
                .filter(s -> s.length() > 0).collect(Collectors.toCollection(ArrayList::new));
    }

    public static CompilationUnit parseFileWithRetries(String code) throws IOException {
        final String classPrefix = "public class Test {";
        final String classSuffix = "}";
        final String methodPrefix = "SomeUnknownReturnType f() {";
        final String methodSuffix = "return noSuchReturnValue; }";

        String originalContent = code;
        String content = originalContent;
        CompilationUnit parsed = null;
        try {
            parsed = JavaParser.parse(content);
        } catch (ParseProblemException e0) {
            try {
                Pattern r = Pattern.compile("#set\\(.+\\)");
                Matcher m = r.matcher(content);
                content = m.replaceAll("");
                parsed = JavaParser.parse(content);
            } catch (ParseProblemException e1) {
                System.out.print(e1);
                // Wrap with a class and method
                try {
                    content = classPrefix + methodPrefix + originalContent + methodSuffix + classSuffix;
                    parsed = JavaParser.parse(content);
                } catch (ParseProblemException e2) {
                    // Wrap with a class only
                    content = classPrefix + originalContent + classSuffix;
                    parsed = JavaParser.parse(content);
                }
            }
        }

        return parsed;
    }

    public static int getBeginLineNumber(Node node) {
        return node.getBegin().line;
    }

    public static String getClassName(Class<?> cls) {
        String[] list = cls.toString().split("\\.");
        return list[list.length - 1];
    }
}
