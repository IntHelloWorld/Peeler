package pathextractor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.PackageDeclaration;
import com.github.javaparser.ast.imports.ImportDeclaration;

import pathextractor.Visitors.ImportExprVisitor;
import pathextractor.Visitors.PakageDeclarationVisitor;

public class ImportClassExtractor {

    public static ArrayList<Path> extractImportClassPaths(CompilationUnit compilationUnit, Path filePath)
            throws IOException {
        Path sourcePath = Paths.get("");
        Pattern r = Pattern.compile("(.*src.test.java)|(.*src.it.java)|(.*src.test)|(.*tests)|(.*test)");
        Matcher m = r.matcher(filePath.toString());
        m.find();
        if (m.group(1) != null) {
            sourcePath = Paths.get(m.group(1).replace("test", "main"));
        } else if (m.group(2) != null) {
            sourcePath = Paths.get(m.group(2).replace("it", "main"));
        } else if (m.group(3) != null) {
            sourcePath = Paths.get(m.group(3).replace("tests", ""));
        } else if (m.group(4) != null) {
            sourcePath = Paths.get(m.group(4).replace("javatests", "java"));
        } else if (m.group(5) != null) {
            sourcePath = Paths.get(m.group(5).replace("test", "src"));
        } else {
            System.out.println("[ERROR] Did match sourcePath!");
        }
        Path newSourcePath = sourcePath;
        ImportExprVisitor importExprVisitor = new ImportExprVisitor();
        importExprVisitor.visit(compilationUnit, null);
        ArrayList<ImportDeclaration> importDeclarations = importExprVisitor.getImportDeclarations();
        ArrayList<Path> importClassPaths = (ArrayList<Path>) importDeclarations.stream()
                .map(f -> getImportClassPath(newSourcePath, f)).filter(f -> f != null).collect(Collectors.toList());

        PakageDeclarationVisitor pakageDeclarationVisitor = new PakageDeclarationVisitor();
        pakageDeclarationVisitor.visit(compilationUnit, null);
        PackageDeclaration packageDeclaration = pakageDeclarationVisitor.getPakageDeclaration();
        if (packageDeclaration != null) {
            importClassPaths.addAll(getPakageClassPath(newSourcePath, packageDeclaration));
        }
        return importClassPaths;
    }

    public static Path getImportClassPath(Path sourcePath, ImportDeclaration importDeclaration) {
        // TODO: different kinds of importDeclarations.

        String importString = importDeclaration.toString();
        Pattern r1 = Pattern.compile("(import\\s+)(.*);");
        Matcher m1 = r1.matcher(importString);
        if (!m1.find()) {
            return null;
        }
        String tmp = m1.group(2);

        String postfix = tmp.replace('.', '/') + ".java";
        Path newPath = Paths.get(sourcePath.toString(), postfix);
        if (Files.notExists(newPath)) {
            return null;
        }
        return newPath;
    }

    public static ArrayList<Path> getPakageClassPath(Path sourcePath, PackageDeclaration packageDeclaration)
            throws IOException {
        String packageString = packageDeclaration.toString();
        Pattern r1 = Pattern.compile("(package\\s+)(.*);");
        Matcher m1 = r1.matcher(packageString);
        if (!m1.find()) {
            return new ArrayList<>();
        }
        String tmp = m1.group(2);
        Path packagePath = Paths.get(sourcePath.toString(), tmp.replace('.', '/'));
        if (Files.notExists(packagePath)) {
            return new ArrayList<>();
        }
        ArrayList<Path> importClassPaths = (ArrayList<Path>) Files.list(packagePath).filter(f -> isJavaFile(f))
                .collect(Collectors.toList());
        return importClassPaths;
    }

    public static boolean isJavaFile(Path path) {
        if (Files.isDirectory(path))
            return false;
        else {
            String filename = path.getFileName().toString();
            String suffix = filename.substring(filename.indexOf('.') + 1);
            if (suffix.equals("java"))
                return true;
            return false;
        }
    }
}
