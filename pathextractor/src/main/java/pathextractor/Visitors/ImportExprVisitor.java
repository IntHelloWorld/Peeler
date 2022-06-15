package pathextractor.Visitors;

import java.util.ArrayList;

import com.github.javaparser.ast.imports.ImportDeclaration;
import com.github.javaparser.ast.imports.SingleStaticImportDeclaration;
import com.github.javaparser.ast.imports.SingleTypeImportDeclaration;
import com.github.javaparser.ast.imports.StaticImportOnDemandDeclaration;
import com.github.javaparser.ast.imports.TypeImportOnDemandDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

public class ImportExprVisitor extends VoidVisitorAdapter<Object> {
    private ArrayList<ImportDeclaration> importDeclarations = new ArrayList<>();

    @Override
    public void visit(SingleTypeImportDeclaration node, Object arg) {
        visitMethod(node, arg);
        super.visit(node, arg);
    }

    @Override
    public void visit(SingleStaticImportDeclaration node, Object arg) {
        visitMethod(node, arg);
        super.visit(node, arg);
    }

    @Override
    public void visit(StaticImportOnDemandDeclaration node, Object arg) {
        visitMethod(node, arg);
        super.visit(node, arg);
    }

    @Override
    public void visit(TypeImportOnDemandDeclaration node, Object arg) {
        visitMethod(node, arg);
        super.visit(node, arg);
    }

    private void visitMethod(ImportDeclaration node, Object arg) {
        importDeclarations.add(node);
    }

    public ArrayList<ImportDeclaration> getImportDeclarations() {
        return importDeclarations;
    }
}
